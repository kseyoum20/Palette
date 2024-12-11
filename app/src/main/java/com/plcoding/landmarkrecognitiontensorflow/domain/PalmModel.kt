package com.plcoding.landmarkrecognitiontensorflow.domain

import android.content.Context
import android.graphics.*
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files.exists
import kotlin.math.max
import kotlin.math.sqrt

class PalmModel(
    modelFileName: String,
    anchorsFileName: String,
    private val boxEnlarge: Float = 1.5f,
    private val boxShift: Float = 0.2f,
    private val context: Context
) {
    private val interpreter: Interpreter
    private val inputSize = 192  // Changed to match model's expected input
    private val anchors: Array<FloatArray>
    private var lastMaskedImagePath: String? = null

    init {
        val assetFileDescriptor = context.assets.openFd(modelFileName)
        val fileInputStream = assetFileDescriptor.createInputStream()
        val modelData = fileInputStream.readBytes()
        fileInputStream.close()

        val byteBuffer = ByteBuffer.allocateDirect(modelData.size)
        byteBuffer.order(ByteOrder.nativeOrder())
        byteBuffer.put(modelData)

        interpreter = Interpreter(byteBuffer)
        anchors = loadAnchors(context, anchorsFileName)
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Calculate padding to make the image square
        val maxDim = maxOf(bitmap.width, bitmap.height)
        val scale = 256f / maxDim

        // Scale dimensions
        val scaledWidth = (bitmap.width * scale).toInt()
        val scaledHeight = (bitmap.height * scale).toInt()

        // Create padded bitmap
        val paddedBitmap = Bitmap.createBitmap(256, 256, bitmap.config)
        val canvas = Canvas(paddedBitmap)
        canvas.drawColor(Color.BLACK)

        // Scale original bitmap
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)

        // Draw scaled bitmap centered
        val left = (256 - scaledWidth) / 2f
        val top = (256 - scaledHeight) / 2f
        canvas.drawBitmap(scaledBitmap, left, top, null)

        // Convert to normalized float buffer (-1 to 1 range)
        val byteBuffer = ByteBuffer.allocateDirect(256 * 256 * 3 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(256 * 256)
        paddedBitmap.getPixels(pixels, 0, 256, 0, 0, 256, 256)

        for (pixel in pixels) {
            // Normalize to [-1, 1] following Python's 2 * (x/255 - 0.5)
            byteBuffer.putFloat(((pixel shr 16 and 0xFF) / 127.5f) - 1f)
            byteBuffer.putFloat(((pixel shr 8 and 0xFF) / 127.5f) - 1f)
            byteBuffer.putFloat(((pixel and 0xFF) / 127.5f) - 1f)
        }
        byteBuffer.rewind()

        return byteBuffer
    }

    fun detectAndVisualize(bitmap: Bitmap): Pair<List<BoundingBox>, Bitmap> {
        val boxes = detect(bitmap)
        val visualizedBitmap = drawBoundingBoxes(bitmap, boxes)
        return Pair(boxes, visualizedBitmap)
    }

    fun detectAndMask(bitmap: Bitmap): Pair<List<BoundingBox>, Bitmap> {
        val boxes = detect(bitmap)
        val maskedBitmap = createMaskedBitmap(bitmap, boxes)
        return Pair(boxes, maskedBitmap)
    }
    // Add target triangle and box like Python
    private val targetTriangle = arrayOf(
        floatArrayOf(128f, 128f),
        floatArrayOf(128f, 0f),
        floatArrayOf(0f, 128f)
    )

    private val targetBox = arrayOf(
        floatArrayOf(0f, 0f, 1f),
        floatArrayOf(256f, 0f, 1f),
        floatArrayOf(256f, 256f, 1f),
        floatArrayOf(0f, 256f, 1f)
    )

    // Add rotation matrix for triangle calculation
    private val r90 = arrayOf(
        floatArrayOf(0f, 1f),
        floatArrayOf(-1f, 0f)
    )

    private fun getTriangle(kp0: FloatArray, kp2: FloatArray, dist: Float = 1f): Array<FloatArray> {
        // Calculate direction vector
        val dirV = floatArrayOf(kp2[0] - kp0[0], kp2[1] - kp0[1])
        // Normalize
        val length = sqrt(dirV[0] * dirV[0] + dirV[1] * dirV[1])
        dirV[0] /= length
        dirV[1] /= length

        // Rotate direction vector
        val dirVR = floatArrayOf(
            dirV[0] * r90[0][0] + dirV[1] * r90[0][1],
            dirV[0] * r90[1][0] + dirV[1] * r90[1][1]
        )

        return arrayOf(
            kp2,
            floatArrayOf(kp2[0] + dirV[0] * dist, kp2[1] + dirV[1] * dist),
            floatArrayOf(kp2[0] + dirVR[0] * dist, kp2[1] + dirVR[1] * dist)
        )
    }


    private fun detect(bitmap: Bitmap): List<BoundingBox> {
        val inputTensor = preprocessImage(bitmap)

        val outputShapeBoxes = interpreter.getOutputTensor(0).shape()
        val outputShapeClf = interpreter.getOutputTensor(1).shape()
        Log.d("PalmModel", "Shapes - boxes: ${outputShapeBoxes.contentToString()}, clf: ${outputShapeClf.contentToString()}")

        val numPredictions = outputShapeBoxes[1]
        val numAttributes = outputShapeBoxes[2]

        // Setup outputs exactly like Python
        val outputTensorBoxes = Array(1) { Array(numPredictions) { FloatArray(numAttributes) } }
        val outputTensorClf = Array(1) { Array(numPredictions) { FloatArray(1) } }

        // Run model
        interpreter.runForMultipleInputsOutputs(
            arrayOf(inputTensor),
            mapOf(0 to outputTensorBoxes, 1 to outputTensorClf)
        )

        // Extract outputs exactly like Python
        val outReg = outputTensorBoxes[0] // [2016, 18]
        val outClf = FloatArray(numPredictions)
        for (i in 0 until numPredictions) {
            outClf[i] = outputTensorClf[0][i][0]
        }

        // Log raw outputs for first few predictions
        for (i in 0..5) {
            Log.d("PalmModel", """
            Prediction $i:
            Regression: ${outReg[i].take(4)}
            Classifier: ${outClf[i]}
            Sigmoid: ${sigmoid(outClf[i])}
        """.trimIndent())
        }

        // Check if model outputs are in valid ranges
        val regMin = outReg.map { it.min() }.min()
        val regMax = outReg.map { it.max() }.max()
        val clfMin = outClf.min()
        val clfMax = outClf.max()

        Log.d("PalmModel", """
        Value ranges:
        Regression: $regMin to $regMax
        Classifier: $clfMin to $clfMax
    """.trimIndent())

        // Process detections
        val probabilities = outClf.map { sigmoid(it) }
        val detectionMask = probabilities.mapIndexed { index, prob -> index }
            .filter { probabilities[it] > 0.5f }

        if (detectionMask.isEmpty()) return emptyList()

        val candidateDetect = detectionMask.map { outReg[it] }
        val candidateAnchors = detectionMask.map { anchors[it] }
        val candidateProbs = detectionMask.map { probabilities[it] }

        // Move candidates
        val movedCandidates = candidateDetect.mapIndexed { index, detection ->
            val anchor = candidateAnchors[index]
            FloatArray(4).apply {
                this[0] = detection[0] + (anchor[0] * inputSize)
                this[1] = detection[1] + (anchor[1] * inputSize)
                this[2] = detection[2]
                this[3] = detection[3]
            }
        }

        val selectedIndices = nonMaxSuppression(movedCandidates.toTypedArray(), candidateProbs.toFloatArray())
        if (selectedIndices.isEmpty()) return emptyList()

        // Process selected detection
        val index = selectedIndices[0]
        val detection = candidateDetect[index]
        val anchor = candidateAnchors[index]

        // Get keypoints (7 landmarks from palm detection)
        val centerNoOffset = floatArrayOf(
            anchor[0] * inputSize,
            anchor[1] * inputSize
        )

        val keypoints = mutableListOf<FloatArray>()
        for (i in 4 until 18 step 2) {
            keypoints.add(floatArrayOf(
                centerNoOffset[0] + detection[i],
                centerNoOffset[1] + detection[i + 1]
            ))
        }

        // Calculate transformed box
        val side = maxOf(detection[2], detection[3]) * boxEnlarge
        val source = getTriangle(keypoints[0], keypoints[2], side)

        // Apply box shift
        val shift = floatArrayOf(
            (keypoints[0][0] - keypoints[2][0]) * boxShift,
            (keypoints[0][1] - keypoints[2][1]) * boxShift
        )
        source.forEach { point ->
            point[0] -= shift[0]
            point[1] -= shift[1]
        }

        // Calculate affine transform
        val transform = android.graphics.Matrix()
        transform.setPolyToPoly(
            source.flatMap { listOf(it[0], it[1]) }.toFloatArray(),
            0,
            targetTriangle.flatMap { listOf(it[0], it[1]) }.toFloatArray(),
            0,
            3
        )

        // Calculate inverse transform
        val inverse = android.graphics.Matrix()
        transform.invert(inverse)

        // Transform box corners
        val finalBox = transformBox(targetBox, inverse)

        val scale = maxOf(bitmap.width, bitmap.height).toFloat() / inputSize
        return listOf(BoundingBox(
            x = finalBox[0][0] * scale,
            y = finalBox[0][1] * scale,
            width = (finalBox[1][0] - finalBox[0][0]) * scale,
            height = (finalBox[2][1] - finalBox[1][1]) * scale,
            confidence = candidateProbs[index]
        ))
    }

    private fun transformBox(box: Array<FloatArray>, transform: Matrix): Array<FloatArray> {
        val points = FloatArray(box.size * 2)
        for (i in box.indices) {
            points[i * 2] = box[i][0]
            points[i * 2 + 1] = box[i][1]
        }

        transform.mapPoints(points)

        return Array(box.size) { i ->
            floatArrayOf(points[i * 2], points[i * 2 + 1])
        }
    }

    private fun drawBoundingBoxes(originalBitmap: Bitmap, boxes: List<BoundingBox>): Bitmap {
        val mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        val paint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 8f
            color = Color.GREEN
        }

        boxes.forEach { box ->
            canvas.drawRect(
                box.x,
                box.y,
                box.x + box.width,
                box.y + box.height,
                paint
            )
        }

        return mutableBitmap
    }

    private fun createMaskedBitmap(originalBitmap: Bitmap, boxes: List<BoundingBox>): Bitmap {
        val mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        // Create a black paint for masking everything
        val blackPaint = Paint().apply {
            color = Color.BLACK
            style = Paint.Style.FILL
        }

        // First black out everything
        canvas.drawRect(0f, 0f, mutableBitmap.width.toFloat(), mutableBitmap.height.toFloat(), blackPaint)

        // Then clear the detected hand regions
        val clearPaint = Paint().apply {
            xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC)
        }

        boxes.forEach { box ->
            canvas.drawBitmap(
                originalBitmap,
                Rect(
                    box.x.toInt(),
                    box.y.toInt(),
                    (box.x + box.width).toInt(),
                    (box.y + box.height).toInt()
                ),
                RectF(
                    box.x,
                    box.y,
                    box.x + box.width,
                    box.y + box.height
                ),
                clearPaint
            )
        }
        // Generate unique path for this masked image
        val maskedImagePath = FileUtils.generateMaskedImagePath(context)
        saveProcessedImage(mutableBitmap, FileUtils.getFileNameFromPath(maskedImagePath))
        lastMaskedImagePath = maskedImagePath

        return mutableBitmap
    }
    // Add getter for last saved path
    fun getLastMaskedImagePath(): String? = lastMaskedImagePath

    private fun saveProcessedImage(bitmap: Bitmap, filename: String) {
        try {
            val outputDir = File(context.filesDir, "processed_data").apply {
                if (!exists()) mkdirs()
            }
            val outputFile = File(outputDir, filename)
            FileOutputStream(outputFile).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
        } catch (e: Exception) {
            Log.e("PalmModel", "Error saving processed image", e)
        }
    }


    private fun sigmoid(x: Float): Float = 1.0f / (1.0f + Math.exp(-x.toDouble())).toFloat()

    private fun nonMaxSuppression(
        boxes: Array<FloatArray>,
        probabilities: FloatArray,
        overlapThreshold: Float = 0.3f
    ): List<Int> {
        // Convert boxes from center format to corner format for IoU
        val corners = boxes.map { box ->
            floatArrayOf(
                box[0] - box[2]/2,  // x1 = centerX - width/2
                box[1] - box[3]/2,  // y1 = centerY - height/2
                box[0] + box[2]/2,  // x2 = centerX + width/2
                box[1] + box[3]/2   // y2 = centerY + height/2
            )
        }

        // Get indices sorted by probability
        val indices = probabilities.indices.sortedByDescending { probabilities[it] }
        val keep = mutableListOf<Int>()
        val active = BooleanArray(indices.size) { true }

        indices.forEachIndexed { i, idx ->
            if (!active[i]) return@forEachIndexed

            keep.add(idx)
            val boxI = corners[idx]

            // Compare with remaining boxes
            for (j in (i + 1) until indices.size) {
                if (!active[j]) continue

                val boxJ = corners[indices[j]]

                // Calculate IoU
                val intersection = calculateIntersectionArea(
                    x1min = boxI[0], y1min = boxI[1], x1max = boxI[2], y1max = boxI[3],
                    x2min = boxJ[0], y2min = boxJ[1], x2max = boxJ[2], y2max = boxJ[3]
                )

                val areaI = (boxI[2] - boxI[0]) * (boxI[3] - boxI[1])
                val areaJ = (boxJ[2] - boxJ[0]) * (boxJ[3] - boxJ[1])
                val union = areaI + areaJ - intersection

                if (intersection / union > overlapThreshold) {
                    active[j] = false
                }
            }
        }

        return keep
    }
    private fun calculateIntersectionArea(
        x1min: Float, y1min: Float, x1max: Float, y1max: Float,
        x2min: Float, y2min: Float, x2max: Float, y2max: Float
    ): Float {
        val xmin = maxOf(x1min, x2min)
        val ymin = maxOf(y1min, y2min)
        val xmax = minOf(x1max, x2max)
        val ymax = minOf(y1max, y2max)

        return maxOf(0f, xmax - xmin) * maxOf(0f, ymax - ymin)
    }

    private fun loadAnchors(context: Context, filename: String): Array<FloatArray> {
        val inputStream = context.assets.open(filename)
        val reader = inputStream.bufferedReader()
        val anchors = mutableListOf<FloatArray>()
        reader.forEachLine { line ->
            val values = line.split(",").map { it.toFloat() }
            anchors.add(values.toFloatArray())
        }
        return anchors.toTypedArray()
    }

    data class BoundingBox(
        val x: Float,
        val y: Float,
        val width: Float,
        val height: Float,
        val confidence: Float
    )
}