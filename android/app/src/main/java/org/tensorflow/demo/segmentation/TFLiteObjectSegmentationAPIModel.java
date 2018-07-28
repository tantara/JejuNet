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

package org.tensorflow.demo.segmentation;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.os.Trace;

import org.tensorflow.demo.env.Logger;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Vector;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteObjectSegmentationAPIModel implements Segmentor {
  private static final Logger LOGGER = new Logger();

  // Config values.
  private int inputWidth;
  private int inputHeight;
  private int numClass;
  public Vector<String> labels = new Vector<String>();

  // Pre-allocated buffers.
  private int[] intValues;
  private long[][] pixelClasses;
  protected ByteBuffer imgData = null;

  private Interpreter tfLite;

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   */
  public static Segmentor create(
      final AssetManager assetManager,
      final String modelFilename,
      final String labelFilename,
      final int inputWidth,
      final int inputHeight,
      final int numClass, final int numOutput) throws IOException {
    final TFLiteObjectSegmentationAPIModel d = new TFLiteObjectSegmentationAPIModel();

    d.inputWidth = inputWidth;
    d.inputHeight = inputHeight;
    d.numClass = numClass;

    try {
      d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
      d.tfLite.setNumThreads(4);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    InputStream labelsInput = null;
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    labelsInput = assetManager.open(actualFilename);
    BufferedReader br = null;
    br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      d.labels.add(line);
    }

    // Pre-allocate buffers.
    d.imgData = ByteBuffer.allocateDirect(d.inputWidth*d.inputHeight*3);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputWidth * d.inputHeight];
    d.pixelClasses = new long[1][d.inputHeight*d.inputWidth*numOutput];
    return d;
  }

  private TFLiteObjectSegmentationAPIModel() {}

  public Vector<String> getLabels() {
    return labels;
  }

  public Segmentation segmentImage(final Bitmap bitmap) {
    if (imgData != null) {
      imgData.rewind();
    }

    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("segmentImage");

    Trace.beginSection("preprocessBitmap");
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    for (int j = 0; j < inputHeight; ++j) {
      for (int i = 0; i < inputWidth; ++i) {
        int pixel = intValues[j*inputWidth + i];
        imgData.put((byte) ((pixel >> 16) & 0xFF));
        imgData.put((byte) ((pixel >> 8) & 0xFF));
        imgData.put((byte) (pixel & 0xFF));
      }
    }
    Trace.endSection(); // preprocessBitmap

    // Run the inference call.
    Trace.beginSection("run");
    long startTime = SystemClock.uptimeMillis();
    tfLite.run(imgData, pixelClasses);
    long endTime = SystemClock.uptimeMillis();
    Trace.endSection(); // run

    Trace.endSection(); // segmentImage

    return new Segmentation(
                pixelClasses[0],
                numClass,
                inputWidth, inputHeight, endTime - startTime,
            tfLite.getLastNativeInferenceDurationNanoseconds() / 1000 / 1000);
  }
}
