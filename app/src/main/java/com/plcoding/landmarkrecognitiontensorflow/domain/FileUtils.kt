package com.plcoding.landmarkrecognitiontensorflow.domain

import android.content.Context

// Add this to a new file FileUtils.kt
object FileUtils {
    private const val MASKED_IMAGE_PREFIX = "masked_palm_"
    private const val PROCESSED_IMAGE_PREFIX = "processed_"

    fun generateMaskedImagePath(context: Context): String {
        val timestamp = System.currentTimeMillis()
        return "${context.filesDir}/processed_data/${MASKED_IMAGE_PREFIX}${timestamp}.png"
    }

    fun generateProcessedImagePath(context: Context): String {
        val timestamp = System.currentTimeMillis()
        return "${context.filesDir}/processed_data/${PROCESSED_IMAGE_PREFIX}${timestamp}.png"
    }

    fun getFileNameFromPath(path: String): String {
        return path.substring(path.lastIndexOf("/") + 1)
    }
}