import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Objects;
import javax.swing.*;

public class SmileDetection {
    public static void main(String[] args) throws IOException {
        // Load the OpenPnP OpenCV library
        nu.pattern.OpenCV.loadLocally();

        // Paths to Haar cascades
        String faceCascadePath = Objects.requireNonNull(
                SmileDetection.class.getResource("haarcascade_frontalface_default.xml")
        ).getPath();
        String smileCascadePath = Objects.requireNonNull(
                SmileDetection.class.getResource("haarcascade_smile.xml")
        ).getPath();

        // Load Haar cascade classifiers
        CascadeClassifier faceDetector = new CascadeClassifier(faceCascadePath);
        CascadeClassifier smileDetector = new CascadeClassifier(smileCascadePath);

        if (faceDetector.empty() || smileDetector.empty()) {
            System.out.println("Error loading cascade classifiers");
            return;
        }

        // Open webcam using OpenPnP
        var webcam = new VideoCapture(0);
        if (!webcam.isOpened()) {
            System.out.println("Error: Cannot access webcam");
            return;
        }

        // Create a window to display the video
        JFrame frame = new JFrame("Smile Detection");
        JLabel videoLabel = new JLabel();
        frame.getContentPane().add(videoLabel);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.setVisible(true);

        Mat matFrame = new Mat();

        while (webcam.read(matFrame)) {
            if (matFrame.empty()) {
                System.out.println("Error: Empty frame captured");
                break;
            }

            // Detect faces
            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(matFrame, faces);

            for (Rect face : faces.toArray()) {
                // Draw rectangle around the face
                Imgproc.rectangle(matFrame, face.tl(), face.br(), new Scalar(0, 255, 0), 2);

                // Detect smiles in the face ROI
                Mat faceROI = matFrame.submat(face);
                MatOfRect smiles = new MatOfRect();
                smileDetector.detectMultiScale(faceROI, smiles, 1.8, 20, 0, new Size(30, 30), new Size());

                // Retain only one smile (if multiple detected)
                Rect[] smileArray = smiles.toArray();
                if (smileArray.length > 0) {
                    Rect smile = smileArray[0]; // Use the first detected smile
                    Rect absoluteSmile = new Rect(
                            face.x + smile.x,
                            face.y + smile.y,
                            smile.width,
                            smile.height
                    );

                    // Draw rectangle around the smile
                    Imgproc.rectangle(matFrame, absoluteSmile, new Scalar(255, 0, 0), 2);
                }
            }

            // Convert the Mat to a BufferedImage and display it
            BufferedImage image = matToBufferedImage(matFrame);

            // Scale image to fit the JFrame
            int frameWidth = frame.getWidth();
            int originalWidth = image.getWidth();
            int originalHeight = image.getHeight();
            int scaledHeight = (int) ((double) frameWidth / originalWidth * originalHeight);
            Image scaledImage = image.getScaledInstance(frameWidth, scaledHeight, Image.SCALE_SMOOTH);
            videoLabel.setIcon(new ImageIcon(scaledImage));
            frame.repaint();

            // Exit on pressing 'q'
            if (System.in.available() > 0 && System.in.read() == 'q') {
                break;
            }
        }

        // Release resources
        webcam.release();
        frame.dispose();
    }

    // Utility method to convert Mat to BufferedImage
    private static BufferedImage matToBufferedImage(Mat mat) {
        int type = mat.channels() > 1 ? BufferedImage.TYPE_3BYTE_BGR : BufferedImage.TYPE_BYTE_GRAY;
        byte[] buffer = new byte[mat.channels() * mat.cols() * mat.rows()];
        mat.get(0, 0, buffer);
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        image.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), buffer);
        return image;
    }
}

