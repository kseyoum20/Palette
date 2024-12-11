package com.plcoding.landmarkrecognitiontensorflow.domain

import java.io.FileInputStream
import java.nio.channels.FileChannel
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import android.util.Log
import com.plcoding.landmarkrecognitiontensorflow.ColorDetector


class FingertipModel(
    private val context: Context,
    private val modelFileName: String = "model.tflite"
) {
    private var interpreter: Interpreter
    private val imageSizeX = 224
    private val imageSizeY = 224
    private val pixelSize = 3
    private val outputSize = 2  // x, y coordinates
    private val colorDetector = ColorDetector(context)

    init {
        // Load model
        val modelFile = loadModelFile()
        interpreter = Interpreter(modelFile)
    }

    private fun loadModelFile(): MappedByteBuffer {
        return context.assets.openFd(modelFileName).use { fileDescriptor ->
            FileInputStream(fileDescriptor.fileDescriptor).use { inputStream ->
                inputStream.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fileDescriptor.startOffset,
                    fileDescriptor.declaredLength
                )
            }
        }
    }

    fun detectFingertip(bitmap: Bitmap): Pair<Float, Float> {
        // Prepare input buffer
        val inputBuffer = ByteBuffer.allocateDirect(imageSizeX * imageSizeY * pixelSize * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Load image data into input buffer
        val pixels = IntArray(imageSizeX * imageSizeY)
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, imageSizeX, imageSizeY, true)
        scaledBitmap.getPixels(pixels, 0, imageSizeX, 0, 0, imageSizeX, imageSizeY)

        pixels.forEach { pixel ->
            inputBuffer.putFloat((Color.red(pixel)) / 255.0f)
            inputBuffer.putFloat((Color.green(pixel)) / 255.0f)
            inputBuffer.putFloat((Color.blue(pixel)) / 255.0f)
        }

        // Prepare output buffer
        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 2), DataType.FLOAT32)

        // Run inference
        interpreter.run(inputBuffer, outputBuffer.buffer)

        // Get normalized coordinates
        val coordinates = outputBuffer.floatArray
        return Pair(coordinates[0], coordinates[1])
    }

    fun drawFingertipOnImage(bitmap: Bitmap, normX: Float, normY: Float): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint().apply {
            style = Paint.Style.STROKE
            color = Color.RED
            strokeWidth = 60f
        }
        val paint2 = Paint().apply {
            style = Paint.Style.STROKE
            color = Color.RED
            strokeWidth = 10f
        }

        val x = normX * bitmap.width
        val y = normY * bitmap.height

        // Get average color
//        val avgColor = colorDetector.getAverageColor(bitmap, x.toInt(), y.toInt(), 10)

        // Draw circle
        paint.style = Paint.Style.STROKE
        canvas.drawCircle(x, y, 60f, paint2)

        // Draw RGB text
        paint.apply {
            color = Color.RED
            textSize = 80f
            textAlign = Paint.Align.CENTER
            style = Paint.Style.FILL_AND_STROKE
            strokeWidth = 7f
        }

        val (avgColor, colorName) = colorDetector.getAverageColorWithName(bitmap, x.toInt(), y.toInt(), 10)

        canvas.drawText("$colorName: (${avgColor.first}, ${avgColor.second}, ${avgColor.third})",
            x, y - 70, paint)

//            x, y - 70, paint)
//        canvas.drawText("RGB: (${avgColor.first}, ${avgColor.second}, ${avgColor.third})",
//            x, y - 70, paint)

        return mutableBitmap
    }

    fun close() {
        interpreter.close()
    }
}