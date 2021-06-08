// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <omp.h>
#include <memory>
#include <string>
#include <fstream>
#include <chrono>  // NOLINT


#if defined(__arm__) || defined(__aarch64__)
#include <opencv2/videoio/legacy/constants_c.h>
#endif

#include "model_deploy/common/include/paddle_deploy.h"
#include "model_deploy/common/include/visualize.h"

DEFINE_string(model_filename, "", "Path of det inference model");
DEFINE_string(params_filename, "", "Path of det inference params");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(model_type, "", "model type");
DEFINE_string(image_list, "", "Path of test image file");
DEFINE_bool(use_camera, false, "Infering with Camera");
DEFINE_int32(camera_id, 0, "Camera id");
DEFINE_string(video_path, "", "Path of input video");
DEFINE_bool(show_result, false, "show the result of each frame with a window");
DEFINE_bool(save_result, false, "save the result of each frame to a video");
DEFINE_string(save_dir, "output", "Path to save visualized image");
DEFINE_int32(batch_size, 1, "Batch size of infering");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
DEFINE_int32(gpu_id, 0, "GPU card id");
DEFINE_bool(use_trt, false, "Infering with TensorRT");

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);

  // create model
  std::shared_ptr<PaddleDeploy::Model> model =
        PaddleDeploy::CreateModel(FLAGS_model_type);

  // model init
  model->Init(FLAGS_cfg_file);

  // inference engine init
  PaddleDeploy::PaddleEngineConfig engine_config;
  engine_config.model_filename = FLAGS_model_filename;
  engine_config.params_filename = FLAGS_params_filename;
  engine_config.use_gpu = FLAGS_use_gpu;
  engine_config.gpu_id = FLAGS_gpu_id;
  engine_config.use_trt = FLAGS_use_trt;
  if (FLAGS_use_trt) {
    engine_config.precision = 0;
  }
  model->PaddleEngineInit(engine_config);

  if (FLAGS_video_path == "" & FLAGS_use_camera == false) {
    std::cerr << "--video_path or --use_camera need to be defined" << std::endl;
    return -1;
  }

  // Open video
  cv::VideoCapture capture;
  if (FLAGS_use_camera) {
    capture.open(FLAGS_camera_id);
    if (!capture.isOpened()) {
      std::cout << "Can not open the camera "
                << FLAGS_camera_id << "."
                << std::endl;
      return -1;
    }
  } else {
    capture.open(FLAGS_video_path);
    if (!capture.isOpened()) {
      std::cout << "Can not open the video "
                << FLAGS_video_path << "."
                << std::endl;
      return -1;
    }
  }

  // Create a VideoWriter
  cv::VideoWriter video_out;
  std::string video_out_path;
  if (FLAGS_save_result) {
    // Get video information: resolution, fps
    int video_width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
    int video_height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
    int video_fps = static_cast<int>(capture.get(CV_CAP_PROP_FPS));
    int video_fourcc;
    if (FLAGS_use_camera) {
      video_fourcc = 828601953;
    } else {
      video_fourcc = CV_FOURCC('M', 'J', 'P', 'G');
    }

    if (FLAGS_use_camera) {
      time_t now = time(0);
      video_out_path =
        PaddleDeploy::generate_save_path(FLAGS_save_dir,
                                    std::to_string(now) + ".mp4");
    } else {
      video_out_path =
        PaddleDeploy::generate_save_path(FLAGS_save_dir, FLAGS_video_path);
    }
    video_out.open(video_out_path.c_str(),
                   video_fourcc,
                   video_fps,
                   cv::Size(video_width, video_height),
                   true);
    if (!video_out.isOpened()) {
      std::cout << "Create video writer failed!" << std::endl;
      return -1;
    }
  }

  std::vector<PaddleDeploy::Result> results;
  cv::Mat frame;
  int key;
  auto start_time = std::chrono::steady_clock::now();
  while (capture.read(frame)) {
    if (FLAGS_show_result || FLAGS_use_camera) {
     key = cv::waitKey(1);
     // When pressing `ESC`, then exit program and result video is saved
     if (key == 27) {
       break;
     }
    } else if (frame.empty()) {
      break;
    }
    // Begin to predict
    std::vector<cv::Mat> imgs;
    imgs.push_back(std::move(frame));
    auto start_time1 = std::chrono::steady_clock::now();
    model->Predict(imgs, &results);
    auto start_time2 = std::chrono::steady_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      start_time2 - start_time1).count();
    std::cout << "result ======= cost time:" <<  ms << std::endl;
    std::cout << results[0] << std::endl;
    // Visualize results
    /*
      int labels_size = 0;
      YAML::Node det_config = YAML::LoadFile(FLAGS_cfg_file);
      for (const auto& label : det_config["label_list"]) {
        labels_size += 1;
      }
    */
    int labels_size = 2;  // 需要看模型的配置文件， 有多少个label
    PaddleDeploy::DetResult result = *(results[0].det_result);
    cv::Mat vis_img =
        PaddleDeploy::Visualize(frame, result, labels_size, 0.5);
    if (FLAGS_show_result || FLAGS_use_camera) {
      cv::imshow("video_detector", vis_img);
    }
    if (FLAGS_save_result) {
      video_out.write(vis_img);
    }

    if (FLAGS_save_result) {
      video_out.write(vis_img);
    }
  }
  auto end_time = std::chrono::steady_clock::now();
  auto ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time).count();
  std::cout << "end. total time:" <<  ms2 << std::endl;

  capture.release();
  if (FLAGS_save_result) {
    video_out.release();
    std::cout << "Visualized output saved as " << video_out_path << std::endl;
  }
  if (FLAGS_show_result || FLAGS_use_camera) {
    cv::destroyAllWindows();
  }

  return 0;
}
