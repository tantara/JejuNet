/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.demo.tracking;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.util.Size;

import org.tensorflow.demo.segmentation.Segmentor;
import org.tensorflow.demo.env.Logger;

import java.util.Vector;

/**
 * A tracker wrapping ObjectTracker that also handles non-max suppression and matching existing
 * objects to new detections.
 */
public class OverlayTracker {
  private final Logger logger = new Logger();

  private Bitmap bmp;

  private Size previewSize;

  private int colors[];
  private long[] resultPixels;
  private int[] pixels;
  private Vector<String> lastLabels;
  private Vector<String> labels;

  public OverlayTracker(final Size previewSize, Vector<String> labels) {
    this.previewSize = previewSize;

    int colors[] = new int[21];
    int alpha = 100;
    colors[0] = Color.argb(alpha, 0, 0, 0);
    colors[1] = Color.argb(alpha, 128, 0, 0);
    colors[2] = Color.argb(alpha, 0, 128, 0);
    colors[3] = Color.argb(alpha, 128, 128, 0);
    colors[4] = Color.argb(alpha, 0, 0, 128);
    colors[5] = Color.argb(alpha, 128, 0, 128);
    colors[6] = Color.argb(alpha, 0, 128, 128);
    colors[7] = Color.argb(alpha, 128, 128, 128);
    colors[8] = Color.argb(alpha, 64, 0, 0);
    colors[9] = Color.argb(alpha, 192, 0, 0);
    colors[10] = Color.argb(alpha, 64, 128, 0);
    colors[11] = Color.argb(alpha, 192, 128, 0);
    colors[12] = Color.argb(alpha, 64, 0, 128);
    colors[13] = Color.argb(alpha, 192, 0, 128);
    colors[14] = Color.argb(alpha, 64, 128, 128);
    colors[15] = Color.argb(alpha, 192, 128, 128);
    colors[16] = Color.argb(alpha, 0, 64, 0);
    colors[17] = Color.argb(alpha, 128, 64, 0);
    colors[18] = Color.argb(alpha, 0, 192, 0);
    colors[19] = Color.argb(alpha, 128, 192, 0);
    colors[20] = Color.argb(alpha, 0, 64, 128);
    this.colors = colors;
    this.labels = labels;
    this.lastLabels = new Vector<String>();
  }

  public synchronized void trackResults(final Segmentor.Segmentation result, final long timestamp) {
    logger.i("Processing from %d", timestamp);
    processResults(timestamp, result);
  }

  public synchronized void draw(final Canvas canvas) {
    if(bmp != null) {
      final Matrix matrix = new Matrix();
      float multiplierX = canvas.getWidth()/(float)bmp.getWidth();
      float multiplierY = multiplierX*(float)previewSize.getWidth()/(float)previewSize.getHeight();
      matrix.postScale(multiplierX, multiplierY);
      matrix.postTranslate(0, 0);
      canvas.drawBitmap(bmp, matrix, new Paint(Paint.FILTER_BITMAP_FLAG));
    }
  }

  private void processResults(
          final long timestamp, final Segmentor.Segmentation result) {
    handleSegmentation(timestamp, result);
  }

  public Vector<String> getLastLabels() {
    return lastLabels;
  }

  private void handleSegmentation(final long timestamp, final Segmentor.Segmentation potential) {
    // very tricky part
    int width = potential.getWidth();
    int height = potential.getHeight();
    if(bmp == null) {
      bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
    }
    if(pixels == null) {
      pixels = new int[bmp.getHeight()*bmp.getWidth()];
    }
    resultPixels = potential.getPixels();

    int numClass = potential.getNumClass();
    int[] visitedLabels = new int[numClass];
    for(int i = 0; i < width; i++) {
      for(int j = 0; j < height; j++) {
        int classNo = (int)resultPixels[j*height+i]; // very tricky part
        pixels[j*bmp.getWidth()+i] = colors[classNo];
        visitedLabels[classNo] = 1;
      }
    }

    lastLabels.clear();
    for(int i = 0; i < numClass; i++) {
      if(visitedLabels[i] == 1) {
        lastLabels.add(labels.get(i));
      }
    }

    bmp.setPixels(pixels, 0, bmp.getWidth(), 0, 0, bmp.getWidth(), bmp.getHeight());
  }
}
