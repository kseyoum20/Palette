package com.plcoding.landmarkrecognitiontensorflow.domain


import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.util.Log
import java.io.File
import java.io.FileOutputStream

class ImagePreprocessor(private val context: Context) {  // Added private val for context
    fun preprocessImage(imagePath: String, outputFileName: String): File? {
        try {
            // Load the image
            val inputBitmap = BitmapFactory.decodeFile(imagePath)
            if (inputBitmap == null) {
                Log.e("ImagePreprocessor", "Failed to load image: $imagePath")
                return null
            }

            // Calculate aspect ratio preserving dimensions
            val targetSize = 224
            val (scaledWidth, scaledHeight) = calculateDimensions(
                inputBitmap.width,
                inputBitmap.height,
                targetSize
            )

            // Create a square bitmap with padding
            val paddedBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(paddedBitmap)

            // Fill with black background
            canvas.drawColor(Color.BLACK)

            // Calculate padding to center the image
            val xOffset = (targetSize - scaledWidth) / 2f
            val yOffset = (targetSize - scaledHeight) / 2f

            // Scale the original image
            val scaledBitmap = Bitmap.createScaledBitmap(
                inputBitmap,
                scaledWidth,
                scaledHeight,
                true
            )

            // Draw the scaled image centered on the padded bitmap
            canvas.drawBitmap(scaledBitmap, xOffset, yOffset, null)

            // Save the preprocessed image
            val outputDir = File(context.filesDir, "processed_data")
            if (!outputDir.exists()) {
                outputDir.mkdir()  // Changed mkdirs() to mkdir()
            }

            val outputFile = File(outputDir, outputFileName)
            FileOutputStream(outputFile).use { out ->
                paddedBitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }

            // Clean up
            inputBitmap.recycle()
            scaledBitmap.recycle()
            paddedBitmap.recycle()

            return outputFile
        } catch (e: Exception) {
            Log.e("ImagePreprocessor", "Error preprocessing image: $imagePath", e)
            return null
        }
    }

    private fun calculateDimensions(width: Int, height: Int, targetSize: Int): Pair<Int, Int> {
        val aspectRatio = width.toFloat() / height.toFloat()
        return if (width > height) {
            // Landscape
            Pair(targetSize, (targetSize / aspectRatio).toInt())
        } else {
            // Portrait or square
            Pair((targetSize * aspectRatio).toInt(), targetSize)
        }
    }
}