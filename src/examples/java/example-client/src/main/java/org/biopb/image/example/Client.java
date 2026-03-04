package org.biopb.image.example;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.TimeUnit;

import javax.imageio.ImageIO;

import com.google.protobuf.ByteString;

import biopb.image.BinData;
import biopb.image.DetectionRequest;
import biopb.image.DetectionResponse;
import biopb.image.DetectionSettings;
import biopb.image.ObjectDetectionGrpc;
import biopb.image.Pixels;
import biopb.image.Point;
import biopb.image.ScoredROI;
import biopb.image.ImageData;
import io.grpc.ChannelCredentials;
import io.grpc.Grpc;
import io.grpc.ManagedChannel;
import io.grpc.TlsChannelCredentials;

public class Client {
	static final String host = "lacss.biopb.org";

	static DetectionRequest buildRequest(BufferedImage image) {
		int width = image.getWidth();
		int height = image.getHeight();
		ByteBuffer byteBuffer = ByteBuffer.allocate(width * height * 3); // 3 for R, G, B

		// Get all pixel data and store in ByteBuffer
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int rgb = image.getRGB(x, y);
				byte r = (byte) ((rgb >> 16) & 0xff); // Red
				byte g = (byte) ((rgb >> 8) & 0xff);  // Green
				byte b = (byte) (rgb & 0xff);         // Blue
				byteBuffer.put(r);
				byteBuffer.put(g);
				byteBuffer.put(b);
			}
		}
		byteBuffer.flip();

		// serialize
		BinData bindata = BinData.newBuilder()
				.setData(ByteString.copyFrom(byteBuffer.array()))
				.build();

		Pixels pixels = Pixels.newBuilder()
				.setDimensionOrder("CXYZT")
				.setBindata(bindata)
				.setDtype("u1")
				.setSizeX(width)
				.setSizeY(height)
				.setSizeC(3)
				.build();

		ImageData imageData = ImageData.newBuilder()
				.setPixels(pixels)
				.build();
	
		DetectionSettings settings = DetectionSettings.newBuilder()
				.setScalingHint((float)1.0)
				.build();

		DetectionRequest request = DetectionRequest.newBuilder()
				.setImageData(imageData)
				.setDetectionSettings(settings)
				.build();

		return request;

	}

	static DetectionResponse runDetection(DetectionRequest request) throws InterruptedException {
		ChannelCredentials cred = TlsChannelCredentials.create();
		ManagedChannel channel = Grpc.newChannelBuilder(host, cred)
				.build();

		ObjectDetectionGrpc.ObjectDetectionBlockingStub stub = ObjectDetectionGrpc.newBlockingStub(channel)
				.withWaitForReady();

		DetectionResponse response = stub.runDetection(request);

		if (channel != null) {
			channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
		}

		return response;
	}

	static void showResult(DetectionResponse response) {
		System.out.printf("Detected %d cells.\n", response.getDetectionsCount());

		int cnt = 1;
		for (ScoredROI scoredRoi : response.getDetectionsList()) {

			System.out.printf("Cell %d - score:%f\n", cnt ++ , scoredRoi.getScore());

			for (Point point : scoredRoi.getRoi().getPolygon().getPointsList() ) {
				System.out.printf("(%.1f-%.1f)", point.getX(), point.getY());
			}
			System.out.printf("\n");
		}
	}


	public static void main(String[] args) throws IOException, InterruptedException {

		BufferedImage image = ImageIO.read(new File(args[0]));

		showResult( runDetection( buildRequest(image) ) );
    }

}
