package com.plcoding.landmarkrecognitiontensorflow

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import com.plcoding.landmarkrecognitiontensorflow.domain.ImagePreprocessor
import com.plcoding.landmarkrecognitiontensorflow.domain.FingertipModel
import android.Manifest
import android.content.pm.PackageManager
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageProxy
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.compose.ui.Alignment
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.plcoding.landmarkrecognitiontensorflow.presentation.CameraPreview
import android.graphics.Matrix
import androidx.camera.core.*

class MainActivity : ComponentActivity() {
    private lateinit var fingertipModel: FingertipModel
    private lateinit var imagePreprocessor: ImagePreprocessor

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize models

        fingertipModel = FingertipModel(
            context = this,
            modelFileName = "model.tflite"
        )

        imagePreprocessor = ImagePreprocessor(this)

        if(!hasRequiredPermissions()) {
            ActivityCompat.requestPermissions(
                this, CAMERAX_PERMISSIONS, 0
            )
        }

        // Load test image
//        val inputStream = assets.open("271.JPG")
//        val originalBitmap = BitmapFactory.decodeStream(inputStream)
//        inputStream.close()

        setContent {
            var currentScreen by remember { mutableStateOf("camera") }
            var processedImage by remember { mutableStateOf<Bitmap?>(null) }
            val controller = remember {
                LifecycleCameraController(applicationContext).apply {
                    setEnabledUseCases(
                        CameraController.IMAGE_CAPTURE or
                                CameraController.VIDEO_CAPTURE
                    )
                }
            }


//            var currentBitmap by remember { mutableStateOf(originalBitmap) }
            when (currentScreen) {
                "camera" -> {
                    Box(modifier = Modifier.fillMaxSize()) {
                        CameraPreview(
                            controller = controller,
                            modifier = Modifier.fillMaxSize()
                        )

                        Button(
                            onClick = {
                                takePhoto(controller) { bitmap ->
                                    val (normX, normY) = fingertipModel.detectFingertip(bitmap)
                                    val finalBitmap = fingertipModel.drawFingertipOnImage(
                                        bitmap,
                                        normX,
                                        normY
                                    )
                                    processedImage = finalBitmap
                                    currentScreen = "preview"
                                }

                            },
                            modifier = Modifier
                                .align(Alignment.BottomCenter)
                                .padding(16.dp)
                        ) {
                            Text("Detect Fingertip")
                        }
                    }
                }
                "preview" -> {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(16.dp)
                    ) {
                        processedImage?.let { bitmap ->
                            Image(
                                bitmap = bitmap.asImageBitmap(),
                                contentDescription = "Detection result",
                                modifier = Modifier
                                    .weight(1f)
                                    .fillMaxWidth()
                            )
                        }

                        Button(
                            onClick = {
                                currentScreen = "camera"
                                processedImage = null
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Take Another Photo")
                        }
                    }
                }
            }
        }
    }

    private fun takePhoto(
        controller: LifecycleCameraController,
        onPhotoTaken: (Bitmap) -> Unit
    ) {
        controller.takePicture(
            ContextCompat.getMainExecutor(applicationContext),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    super.onCaptureSuccess(image)
                    val matrix = Matrix().apply {
                        postRotate(image.imageInfo.rotationDegrees.toFloat())
                    }
                    val rotatedBitmap = Bitmap.createBitmap(
                        image.toBitmap(),
                        0,
                        0,
                        image.width,
                        image.height,
                        matrix,
                        true
                    )
                    onPhotoTaken(rotatedBitmap)
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("Camera", "Couldn't take photo: ", exception)
                }
            }
        )
    }


    private fun hasRequiredPermissions(): Boolean {
        return CAMERAX_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(
                applicationContext,
                it
            ) == PackageManager.PERMISSION_GRANTED
        }
    }

    companion object {
        private val CAMERAX_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        fingertipModel.close()
    }
}