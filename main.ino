#include <opencv2/opencv.hpp>
#include <mediapipe/mediapipe.h>
#include <mediapipe/framework/formats/landmark.pb.h>
#include <mediapipe/modules/hand_landmark/hand_landmark.pb.h>

int main() {
    // Set up OpenCV video capture
    cv::VideoCapture video(0);
    if (!video.isOpened()) {
        std::cerr << "Could not open video stream." << std::endl;
        return -1;
    }

    // Set up MediaPipe hand detection
    mediapipe::CalculatorGraph graph;
    std::string graph_config = R"(
        input_stream: "input_video"
        output_stream: "output_landmarks"
        node {
            calculator: "HandLandmarkSubgraph"
            input_stream: "IMAGE:input_video"
            output_stream: "LANDMARKS:output_landmarks"
        })";

    // Load the graph configuration
    graph.Initialize(graph_config);

    // Start the graph
    graph.StartRun({});

    // Initialize variables for hand landmark detection
    std::vector<int> tipIds = {4, 8, 12, 16, 20}; // Thumb, index, middle, ring, little
    cv::Mat frame;
    std::vector<mediapipe::NormalizedLandmarkList> hand_landmarks;

    while (true) {
        // Capture frame
        video >> frame;
        if (frame.empty()) {
            std::cerr << "Frame capture failed." << std::endl;
            break;
        }

        // Convert the frame to RGB
        cv::Mat rgb_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

        // Create a MediaPipe image from the OpenCV frame
        mediapipe::ImageFrame input_frame(mediapipe::ImageFormat::SRGB, rgb_frame.cols, rgb_frame.rows, mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
        std::memcpy(input_frame.MutableData(), rgb_frame.data, rgb_frame.total() * rgb_frame.elemSize());

        // Send the frame to the MediaPipe graph
        graph.AddPacketToInputStream("input_video", mediapipe::Adopt(&input_frame).At(mediapipe::Timestamp::PostStream()));

        // Get the hand landmarks output
        auto& output_packet = graph.GetOutputStream("output_landmarks").Value().Value();
        if (output_packet.IsEmpty()) {
            std::cerr << "No landmarks detected." << std::endl;
            continue;
        }

        // Parse the output landmarks
        const auto& landmarks = output_packet.Get<mediapipe::LandmarkList>();
        hand_landmarks = landmarks.landmark();

        // Detect fingers and count them
        std::vector<int> fingers;
        int total = 0;
        if (!hand_landmarks.empty()) {
            const auto& landmarks = hand_landmarks[0].landmark();

            if (landmarks[tipIds[0]].x > landmarks[tipIds[0] - 1].x) {
                fingers.push_back(1); // Thumb extended
            } else {
                fingers.push_back(0);
            }

            for (int i = 1; i <= 4; ++i) {
                if (landmarks[tipIds[i]].y < landmarks[tipIds[i] - 2].y) {
                    fingers.push_back(1); // Finger extended
                } else {
                    fingers.push_back(0);
                }
            }

            total = std::count(fingers.begin(), fingers.end(), 1);
        }

        // Display the result on the frame
        cv::putText(frame, "Fingers: " + std::to_string(total), cv::Point(45, 375), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 5);

        // Show the output frame
        cv::imshow("Frame", frame);

        // Exit on 'q' key press
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release video capture and clean up
    video.release();
    cv::destroyAllWindows();
    return 0;
}
