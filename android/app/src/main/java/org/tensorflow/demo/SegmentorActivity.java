/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import org.tensorflow.demo.common.OverlayView;
import org.tensorflow.demo.common.OverlayView.DrawCallback;
import org.tensorflow.demo.common.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.segmentation.Segmentor;
import org.tensorflow.demo.segmentation.TFLiteObjectSegmentationAPIModel;
import org.tensorflow.demo.tracking.OverlayTracker;
import me.tantara.real_time_segmentation.R;

import java.io.IOException;
import java.util.Vector;

/**
 * An activity that uses a TensorFlowMultiBoxsegmentor and ObjectTracker to segment and then track
 * objects.
 */
public class SegmentorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged DeepLab model.
  private static final int TF_OD_API_INPUT_WIDTH = 256;
  private static final int TF_OD_API_INPUT_HEIGHT = 256;
  private static final int TF_OD_API_NUM_CLASS = 21;
  private static final int TF_OD_API_NUM_OUTPUT = 1;
  private static final String TF_OD_API_MODEL_FILE = "mobilenet_v2_deeplab_v3_256_myquant.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/pascal_voc_labels_list.txt";

  private enum segmentorMode {
    TF_OD_API;
  }

  private static final segmentorMode MODE = segmentorMode.TF_OD_API;

  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Segmentor segmentor;

  private long lastProcessingTimeMs;
  private long lastInferenceTimeMs;
  private long lastNativeTimeMs;
  private Vector<String> lastLabels;

  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingSegmentation = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private OverlayTracker tracker;

  private BorderedText borderedText;
  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
            TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    try {
      segmentor =
              TFLiteObjectSegmentationAPIModel.create(
                      getAssets(),
                      TF_OD_API_MODEL_FILE,
                      TF_OD_API_LABELS_FILE,
                      TF_OD_API_INPUT_WIDTH,
                      TF_OD_API_INPUT_HEIGHT,
                      TF_OD_API_NUM_CLASS,
                      TF_OD_API_NUM_OUTPUT);
      ImageView overlay = findViewById(R.id.overlay);
      overlay.setVisibility(View.VISIBLE);

      tracker = new OverlayTracker(DESIRED_PREVIEW_SIZE, segmentor.getLabels());
    } catch (final IOException e) {
      LOGGER.e("Exception initializing classifier!", e);
      Toast toast = Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    int cropHeight = TF_OD_API_INPUT_HEIGHT;
    int cropWidth = TF_OD_API_INPUT_WIDTH;

    sensorOrientation = rotation - getScreenOrientation();
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropWidth, cropHeight, Config.ARGB_8888);

    frameToCropTransform = ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropWidth, cropHeight,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    segmentOverlay = (OverlayView) findViewById(R.id.segment_overlay);
    segmentOverlay.addCallback(
            new DrawCallback() {
              @Override
              public void drawCallback(final Canvas canvas) {
                tracker.draw(canvas);
              }
            });

    addCallback(
            new DrawCallback() {
              @Override
              public void drawCallback(final Canvas canvas) {
                if(lastLabels == null) {
                  return;
                }

                final Vector<String> lines = new Vector<String>();
                lines.add("Project:");
                lines.add("- Title: Real-Time Segmentation on Mobile Devices");
                lines.add("- Code: github.com/tantara/JejuNet");
                lines.add("");
                lines.add("Info:");
                lines.add("- TF Lite(Native) Time: " + lastNativeTimeMs + "ms");
                lines.add("- TF Lite(Java) Overhead: " + (lastInferenceTimeMs - lastNativeTimeMs) + "ms");
                lines.add("- Pre/Post Processing Overhead: " + (lastProcessingTimeMs - lastInferenceTimeMs) + "ms");
                lines.add("- Labels: " + String.join(", ", lastLabels));
                lines.add("");
                lines.add("Thanks everyone at Deep Learning Camp Jeju!");
                borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
              }
            });
  }

  OverlayView segmentOverlay;

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    segmentOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingSegmentation) {
      readyForNextImage();
      return;
    }
    computingSegmentation = true;
    LOGGER.i("Preparing image " + currTimestamp + " for segmention in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
            new Runnable() {
              @Override
              public void run() {
                LOGGER.i("Running segmention on image " + currTimestamp);

                final long startTime = SystemClock.uptimeMillis();
                final Segmentor.Segmentation result = segmentor.segmentImage(croppedBitmap);
                lastInferenceTimeMs = result.getInferenceTime();
                lastNativeTimeMs = result.getNativeTime();
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                tracker.trackResults(result, currTimestamp);
                lastLabels = tracker.getLastLabels();
                cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);

                segmentOverlay.postInvalidate();
                requestRender();
                computingSegmentation = false;
              }
            });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_segment;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }
}
