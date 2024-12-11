package com.plcoding.landmarkrecognitiontensorflow

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.pow
import kotlin.math.sqrt
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

data class ColorInfo(val name: String, val rgb: List<Int>)


class ColorDetector(private val context: Context) {
    private val colorMap: Map<String, List<Int>> = Gson().fromJson(context.assets.open("colors.json").bufferedReader().use { it.readText() },
        object : TypeToken<Map<String, List<Int>>>() {}.type
    )

    fun getAverageColorWithName(bitmap: Bitmap, centerX: Int, centerY: Int, radius: Int = 10): Pair<Triple<Int, Int, Int>, String> {
        val avgColor = getAverageColor(bitmap, centerX, centerY, radius)
        val colorName = findClosestColor(avgColor)
        return Pair(avgColor, colorName)
    }

    fun getAverageColor(bitmap: Bitmap, centerX: Int, centerY: Int, radius: Int = 10): Triple<Int, Int, Int> {
        var redSum = 0
        var greenSum = 0
        var blueSum = 0
        var pixelCount = 0

        for (x in centerX - radius..centerX + radius) {
            for (y in centerY - radius..centerY + radius) {
                if (x >= 0 && x < bitmap.width && y >= 0 && y < bitmap.height) {
                    val pixel = bitmap.getPixel(x, y)
                    redSum += Color.red(pixel)
                    greenSum += Color.green(pixel)
                    blueSum += Color.blue(pixel)
                    pixelCount++
                }
            }
        }

        return Triple(
            redSum / pixelCount,
            greenSum / pixelCount,
            blueSum / pixelCount
        )
    }

    private fun findClosestColor(rgb: Triple<Int, Int, Int>): String {
        return colorMap.minBy { (_, colorRgb) ->
            euclideanDistance(
                rgb,
                Triple(colorRgb[0], colorRgb[1], colorRgb[2])
            )
        }.key
    }

    private fun euclideanDistance(c1: Triple<Int, Int, Int>, c2: Triple<Int, Int, Int>): Double {
        return sqrt(
            (c1.first - c2.first).toDouble().pow(2) +
                    (c1.second - c2.second).toDouble().pow(2) +
                    (c1.third - c2.third).toDouble().pow(2)
        )
    }


}