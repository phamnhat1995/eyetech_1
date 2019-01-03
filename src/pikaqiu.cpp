#include "network.h"
#include "mtcnn.h"
#include <time.h>
#include <mutex>
#include <thread>
#include <chrono>
#include <atomic>
#include "send_data.h"
#include "detect_blury.h"
#include "pre_processing_image.h"
#include <future>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace Runner {
    Mat last_frame;
    Mat temp_frame;

    mutex last_lock;
    atomic_flag last_used;

    VideoCapture cap;
    mtcnn *cnn;
	auto results = vector<future<long>>();
	struct config {
		string email;
		string password;
		string dataURL;
		string endpoint;
		string camera_id;

		config(string inputEmail, string inputPassword, string inputURL, string inputEndpoint, string inputCameraID) {
			email = inputEmail;
			password = inputPassword;
			dataURL = inputURL;
			endpoint = inputEndpoint;
			camera_id = inputCameraID;
		}
	};
	struct dataMat
	{
		std::vector<cv::Mat> arrayMat;
		int id;
		bool isSend;
		dataMat(std::vector<cv::Mat> array, int id_give,bool isSend_give)
		{
			arrayMat = array;
			id = id_give;
			isSend = isSend_give;
		}
	};
	std::vector<dataMat> data;
	std::vector<int> arrID;
	void getDataToBatch();

	Mat preProcesingImage(cv::Mat image)
	{
		cv::Mat afterProcessing = image;
		resize(image, afterProcessing, Size(112, 112));
		//afterProcessing = denoisingImage(afterProcessing);
		//afterProcessing = preprocessingImage(afterProcessing);
		//imshow("TrackingResult", afterProcessing);
		//sendImageData(getEncodedImage(afterProcessing));
		/*int random;
		random = (rand() % 100 + 1);
		cv::imwrite("./test/image" + std::to_string(random) + ".jpg", afterProcessing);*/
		return afterProcessing;
	}

    int init() {
        cap = VideoCapture(0);
		//cap = VideoCapture("vn.mp4");
        Mat image;
        if (!cap.isOpened()) {
            cout << "Failed to open!" << endl; 
            return -1;
        }
        cap >> image;
        if (!image.data){
            cout << "No image data!" << endl;  
            return -1;
        }

        cnn = new mtcnn(image.rows, image.cols);
        cap >> last_frame;
        return 0;
    }

	void called_from_async() {
		getDataToBatch();
	}

	void send_data() {
		const auto timeWindow = std::chrono::milliseconds(5000);
		//cout << "here";

		while (true)
		{
			auto start = std::chrono::steady_clock::now();
			getDataToBatch();
			auto end = std::chrono::steady_clock::now();
			auto elapsed = end - start;

			auto timeToWait = timeWindow - elapsed;
			if (timeToWait > std::chrono::milliseconds::zero())
			{
				std::this_thread::sleep_for(timeToWait);
			}
		}
	}

    void read_camera() {
        while (true) {
            cap >> temp_frame;
            if (!last_lock.try_lock()) continue;
            cv::swap(last_frame, temp_frame);
            last_used.clear();
            last_lock.unlock();
        }
    }
    void run() {
        while (true) {
            last_lock.lock();
            if (!last_used.test_and_set()) {
                cnn->findFace(last_frame);
                imshow("result", last_frame);
                if (waitKey(1) >= 0) break;   
            }
            last_lock.unlock();
        }
    }

    int main() {
        assert(init() == 0);

        thread camera(read_camera);
        thread runner(run);
		thread sendData(send_data);

        //runner.join();
		sendData.~thread();
        camera.~thread();

        return 0;
    }

	bool checkDuplicateIdData(int id)
	{
		//cout << "arrID";
		for (auto & value : arrID) {
			//cout << " " + value;
			if (value == id) {
				return true;
			}
		}
		return false;
	}

	void addToDataMat(cv::Mat image, int id)
	{
		bool found = false;
		if (data.size() == 0) {
			arrID.push_back(id);
			std::vector<cv::Mat> tempArr;
			tempArr.push_back(image);
			dataMat temp(tempArr, id, false);
			data.push_back(temp);
		}
		else {
			for (auto & value : data) {
				if (value.id == id && !value.isSend) {
					//cout << "\n check boolean" << value.isSend << "\n";
					found = true;
					value.arrayMat.push_back(image);
				}
			}
			if (!found && !checkDuplicateIdData(id)) {
				arrID.push_back(id);
				std::vector<cv::Mat> tempArr;
				tempArr.push_back(image);
				dataMat temp(tempArr, id, false);
				data.push_back(temp);
			}
		}
	}

	void printData(vector<Mat> data, int id) {
			//cout << "size: " << data.size() << " ";
			int count = 0;
			for (auto & value : data) {
				count++;
				cv::imwrite("./test/image_" + std::to_string(id) + "_" + std::to_string(count) + ".jpg", value);
			}
	}

	void getDataToBatch()
	{
		if (data.size() == 0) {
			return;
		}
		//cout << "size " << data.size();
		auto it = std::begin(data);
		int i = 0;
		while (it != std::end(data)) {
			// Do some stuff
			if (data.at(i).arrayMat.size() > 14 && !data.at(i).isSend) {
				//cout << "sendata";
				std::string temp = makeArrayImage(data.at(i).arrayMat);
				results.push_back(async(launch::async, [temp]() -> long {
					sendImageData(temp);
					return 0;
				}));
				cout << "Send data id: " << data.at(i).id << "\n";
				cout << "-----------------\n";
				//cout << makeArrayImage(data.at(i).arrayMat;
				data.at(i).isSend = true;
				//cout << "modify Boolean " << data.at(i).id << " " << data.at(i).isSend;
				data.at(i).arrayMat.clear();
				//it = data.erase(it);
			}
			else {
				++it;
				++i;
			}
		}
	}
};

Runner::config getConfigFile() {
	std::ifstream i("../config.json");
	json j;
	i >> j;
	Runner::config config(j["email"].get<std::string>(), j["password"].get<std::string>(), j["dataStream"].get<std::string>(), j["endpoint"].get<std::string>(), j["camera_id"].get<std::string>());
	return config;
}

int old_main() {
	/*thread t;
	t.~thread();*/
	cout << "-----------------\n";
	cout << "Start Program \n";
	cout << "-----------------\n";
	Runner::config config = getConfigFile();
	cout << "Done get config \n";
	cout << "-----------------\n";
	getTokenByRequest(config.email, config.password, config.camera_id, config.endpoint);
	cout << "Done get token \n";
    Size sz = Size(640, 480);

    Mat image;
	VideoCapture cap;

	if (!cap.open(config.dataURL)) {
		cout << "fail to open data!" << endl;
		return -1;
	}
	cap >> image;
	resize(image, image, sz, 0, 0);
    if(!image.data){
        cout<<"fuck"<<endl;  
        return -1;
    }
	cout << "-----------------\n";

	cout << "Done get dataStream \n";
	cout << "-----------------\n";

	cout << image.rows << "x" << image.cols << endl;

    mtcnn find(image.rows, image.cols);

    clock_t start;

    int num_frame = 0;
    double total_time = 0;
    int frame_id = 0;

    SORT sorter(15);

    while (true) {
        //start = clock();
        cap>>image;
        resize(image, image, sz, 0, 0);

        auto start_time = std::chrono::system_clock::now();
        vector<Rect_<float> > boxes = find.findFace(image);
        auto diff1 = std::chrono::system_clock::now() - start_time;
        auto t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(diff1);

        vector<TrackingBox> detFrameData;
        for (int i = 0; i < boxes.size(); ++i) {
            //cerr << boxes[i].x << ' ' << boxes[i].y << ' ' << boxes[i].width << ' ' << boxes[i].height << endl;
            TrackingBox cur_box;
            cur_box.box = boxes[i];
            cur_box.id = i;
            cur_box.frame = frame_id;
            detFrameData.push_back(cur_box);
        }
        ++frame_id;

        auto start_track_time = std::chrono::system_clock::now();
        vector<TrackingBox> tracking_results = sorter.update(detFrameData);
        //vector<TrackingBox> tracking_results = detFrameData;
        auto diff2 = std::chrono::system_clock::now() - start_track_time;
        auto t2 = std::chrono::duration_cast<std::chrono::nanoseconds>(diff2);

		//std::future<void> result(std::async(Runner::called_from_async));
		//result.get();
		Runner::getDataToBatch();
		//Mat cloneTest = image.clone();

        for (TrackingBox it : tracking_results) {
			rectangle(image, Point(it.box.y, it.box.x), Point(it.box.height, it.box.width), sorter.randColor[it.id % 20], 2, 8, 0);
			/*rectangle(cloneTest, Point(it.box.y, it.box.x), Point(it.box.height, it.box.width), sorter.randColor[it.id % 20], 2, 8, 0);
			Mat imageTracking = image(Rect(Point(it.box.y, it.box.x), Point(it.box.height, it.box.width)));
			bool isBlur = detectBlur(imageTracking);
			if (isBlur) {
				continue;
			}
			cv::Mat result = Runner::preProcesingImage(imageTracking);
			Runner::addToDataMat(result, (int)it.id);*/
        }
        
        //cout<<"time is  "<<start/10e3<<endl;
        imshow("result", image);
        if( waitKey(1)>=0 ) break;
        start = clock() -start;
         
        //cerr << num_frame << ' ' << t1.count()/1e6 << ' ' << t2.count()/1e6 << " (ms) " << endl;
        if (num_frame < 100) {
            num_frame += 1;
            total_time += double(t1.count());
            total_time += double(t2.count());
        } else {
            printf("Time=%.2f, Frame=%d, FPS=%.2f\n", total_time / 1e9, num_frame, num_frame * 1e9 / total_time);
            num_frame = 0;
            total_time = 0;
        }
    }

    image.release();

    return 0;
}

int main(int argc, char** argv)
{
    //Mat image = imread("4.jpg");
    //mtcnn find(image.rows, image.cols);
    /*
    clock_t start;
    start = clock();
    find.findFace(image);
    imshow("result", image);
    imwrite("result.jpg",image);
    start = clock() -start;
    cout<<"time is  "<<start/10e3<<endl;*/
    

    //return Runner::main();

    return old_main();
    
    return 0;
}

