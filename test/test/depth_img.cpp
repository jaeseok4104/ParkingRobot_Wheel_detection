#include "slamBase.h"


#include <algorithm>
void depth_delete(cv::Mat &edgeimg, cv::Mat &depth);

int main(void)
{
    RealSense realsense;
    ParameterReader pd;
    cout<<"Realsense loading complete!!"<<endl;
    FRAME currFrame;
    int currIndex = 0;

    int start_x = atoi(pd.getData("start_x").c_str());
    int start_y = atoi(pd.getData("start_y").c_str());
    int end_x = atoi(pd.getData("end_x").c_str());
    int end_y = atoi(pd.getData("end_y").c_str());

    int threshold = atoi(pd.getData("threshold").c_str()); // 이진화 threshold value

    int canny_threshold1 = atoi(pd.getData("canny_threshold1").c_str());
    int canny_threshold2 = atoi(pd.getData("canny_threshold2").c_str());
    
    int dp = atoi(pd.getData("dp").c_str());
    double minDist = atoi(pd.getData("minDist").c_str());
    int minRadius = atoi(pd.getData("minRadius").c_str());
    int maxRadius = atoi(pd.getData("maxRadius").c_str());
    double accumulate_value = atoi(pd.getData("accumulate_value").c_str());
    int searchcircle = atoi(pd.getData("searchcircle").c_str());
    double w = atoi(pd.getData("low_pass_weight").c_str());
    int detect_threshold = atoi(pd.getData("detect_threshold").c_str());

    int lowH = atoi(pd.getData("lowH").c_str());
    int lowS = atoi(pd.getData("lowS").c_str());
    int lowV = atoi(pd.getData("lowV").c_str());
    int highH = atoi(pd.getData("highH").c_str());
    int highS = atoi(pd.getData("highS").c_str());
    int highV = atoi(pd.getData("highV").c_str());

    char detect_circle_cnt = 0;
    vector<cv::Vec3f> pre_circles;

    cv::Mat elements(7,7,CV_8U, cv::Scalar(1));
    while(true)
    {
        cout<<"Reading files "<<currIndex<<endl;
        currFrame = readFrame(realsense);
        FRAME roiFrame, roiFrameGray; // roi이미지 및 roi grayscale 이미지
        cv::Mat binaryimg; // 이진화 이미지
        cv::Mat binaryimg_close;
        cv::Mat edgeimg; // 엣지 이미지
        cv::Mat edgeimg_close; // 엣지 이미지
        cv::Mat hsvimg;   // hsv 이미지
        cv::Mat hsvimg_binary; // hsv 이진이미지
        cv::Mat hsvimg_binaryMopol; // hsv 이진 close 이미지
        cv::Mat hsvimg_binaryblur; // hsv_이진 블러 이미지
        cv::Mat hsvedge; // hsv 엣지
        cv::Mat addEdge;
        cv::Mat addEdgeErode;
        
        roiFrame.rgb = currFrame.rgb(cv::Rect(start_x,start_y,end_x,end_y));      // roi 이미지 업데이트 -> currframe에서 추출
        roiFrame.depth = currFrame.align(cv::Rect(start_x,start_y,end_x,end_y));  // roidepth 이미지 업데이트 -> currframe에서 추출
        cv::cvtColor(roiFrame.rgb, roiFrameGray.rgb,cv::COLOR_BGR2GRAY);          // roi grayscale img 업데이트 -> roiframe에서 추출
        cv::cvtColor(roiFrame.rgb, hsvimg, cv::COLOR_BGR2HSV);                    // roi hsv img 업데이트 -> roiframe에서 추출
        
        cv::inRange(hsvimg, cv::Scalar(lowH,lowS,lowV),cv::Scalar(highH,highS,highV), hsvimg_binary);
        cv::dilate(hsvimg_binary, hsvimg_binaryMopol, cv::Mat());
        cv::erode(hsvimg_binaryMopol, hsvimg_binaryMopol, cv::Mat());

        cv::threshold(roiFrameGray.rgb, binaryimg, threshold, 255, cv::THRESH_BINARY_INV| cv::THRESH_OTSU); // 이진화
        cv::dilate(binaryimg, binaryimg_close, cv::Mat());
        cv::erode(binaryimg_close, binaryimg_close, cv::Mat());

        // cv::blur(binaryimg, edgeimg, cv::Size(3,3)); // 일반 이미지
        cv::blur(binaryimg_close, edgeimg_close, cv::Size(3,3));  // 일반이미지 닫힘연산
        // cv::Canny(edgeimg, edgeimg, canny_threshold1, canny_threshold2, 5); // 일반이미지 캐니엣지
        cv::Canny(edgeimg_close, edgeimg_close, canny_threshold1, canny_threshold2, 5);  //  일반 이미지 닫힘연산 캐니엣지
        depth_delete(edgeimg_close, roiFrame.depth);
        cv::imshow("edgeimg_close", edgeimg_close);  // RGB edge




        cv::imshow("hsv_Mopol_pre", hsvimg_binaryMopol); // HSV
        for (int i = 0; i < hsvimg_binaryMopol.rows; i++)
        {
            uchar *img_ptr = hsvimg_binaryMopol.ptr<uchar>(i);
            uchar *def_ptr = binaryimg_close.ptr<uchar>(i); // RGB
            for (int j = 0; j < hsvimg_binaryMopol.cols; j++)
            {
                if(def_ptr[j] != img_ptr[j])
                    img_ptr[j] = 0;
            }
        }

        depth_delete(hsvimg_binaryMopol, roiFrame.depth);
        cv::blur(hsvimg_binaryMopol, hsvimg_binaryblur, cv::Size(3,3));
        cv::Canny(hsvimg_binaryblur, edgeimg, canny_threshold1, canny_threshold2, 5); // hsv 모폴로지 이진 이미지를 캐니 에지로 연산\


        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;

        cv::Mat contoursRGB = roiFrame.rgb.clone();
        // cv::RNG rng(12345); // random number generator
        cv::findContours(edgeimg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
        // cv::Mat drawing = cv::Mat::zeros(roiFrame.rgb.size(), CV_8UC3);
        int pre_idx[2] = {0,0};

        for(int i = 0; i<contours.size(); i++){
            if(contours[i].size() > pre_idx[0]){
                pre_idx[1] = pre_idx[0];
                pre_idx[0] = i;
            }
            else if(contours[i].size() > pre_idx[1])
                pre_idx[1] = i;
        }
        cv::Scalar color = cv::Scalar(0, 0, 255);
        // cout<<"contour size : "<< contours[pre_idx].size()<<endl;
        cv::drawContours(contoursRGB, contours, pre_idx[0], color, 1, 16, hierarchy, 0, cv::Point());
        cv::drawContours(contoursRGB, contours, pre_idx[1], color, 1, 16, hierarchy, 0, cv::Point());
        cv::imshow("roiFrame", roiFrame.rgb);
        cv::imshow("contours_drawing", contoursRGB);
        
        // cv::RotatedRect tmp = cv::fitEllipse(contours);
        // vector<cv::RotatedRect> ellipse;
        // ellipse.clear();
        // for(int i; i<contours.size(); i++){
        //     if(contours[i].size() >= 5){
        //         cv::RotatedRect tmp = cv::fitEllipse(cv::Mat(contours[i]));
        //         if()
        //     }
        // }



        vector<cv::Vec3f> circles;

        cv::HoughCircles(edgeimg, circles, CV_HOUGH_GRADIENT, dp, minDist, searchcircle, accumulate_value, minRadius, maxRadius);
        cout<<"cicles.size = "<< circles.size()<< endl;
        cv::Mat circle_image = roiFrame.rgb.clone();

        // 가중평균필터
        if(circles.size()>0){
            if(detect_circle_cnt<detect_threshold){
                detect_circle_cnt++;
                pre_circles = circles;
            }
            else{
                circles[0][0] = ((w*circles[0][0])/10) + ((10-w)*pre_circles[0][0]/10);
                circles[0][1] = ((w*circles[0][1])/10) + ((10-w)*pre_circles[0][1]/10);
                circles[0][2] = ((w*circles[0][2])/10) + ((10-w)*pre_circles[0][2]/10);
                for (size_t i = 0;i<circles.size(); i++){
                    cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                    int radius = cvRound(circles[i][2]);

                    cv::circle(circle_image, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
                    cv::circle(circle_image, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
                }
                pre_circles = circles;
            }
        }else detect_circle_cnt = 0;


        // cv::imshow("hsv_Mopol_pre", hsvimg_binaryMopol);
        // for (int i = 0; i < hsvimg_binaryMopol.rows; i++)
        // {
        //     uchar *img_ptr = hsvimg_binaryMopol.ptr<uchar>(i);
        //     uchar *def_ptr = binaryimg_close.ptr<uchar>(i);
        //     for (int j = 0; j < hsvimg_binaryMopol.cols; j++)
        //     {
        //         if(def_ptr[j] != img_ptr[j])
        //             img_ptr[j] = 0;
        //     }
        // }

        // depth_delete(hsvimg_binaryMopol, roiFrame.depth);
        // cv::blur(hsvimg_binaryMopol, hsvimg_binaryblur, cv::Size(3,3));
        // cv::Canny(hsvimg_binaryblur, hsvedge, canny_threshold1, canny_threshold2, 5); // hsv 모폴로지 이진 이미지를 캐니 에지로 연산

        // data check
        cv::imshow("binaryimg_close", binaryimg_close); // RGB
        cv::imshow("hsv", hsvimg); // HSV
        // cv::imshow("hsv_binary", hsvimg_binary);
        cv::imshow("hsv_Mopol", hsvimg_binaryMopol); // AND 연산
        cv::imshow("hsv_binaryblur", hsvimg_binaryblur); // 위에 블러
        // cv::imshow("hsvedge", hsvedge);
        cv::imshow("edgeimg", edgeimg); // 엣지
        cv::imshow("circle", circle_image); //원

        int check = cv::waitKey(1);
        if (check == 's')
        {
        }
        else if (check == 'q')
            break;
        currIndex++;
    }

    return 0;
}

void depth_delete(cv::Mat &edgeimg, cv::Mat &depth)
{
    for (int i = 0; i < edgeimg.rows; i++)
    {
        uchar *img_ptr = edgeimg.ptr<uchar>(i);
        uchar *depth_ptr = depth.ptr<uchar>(i);
        for (int j = 0; j < edgeimg.cols; j++)
        {
            // if (((depth_ptr[j + 5] > 3) && (depth_ptr[j - 5] > 3) && ((depth_ptr - 5)[j] > 3) && ((depth_ptr + 5)[j] > 3)))
            if ((depth_ptr[j] > 2)|| depth_ptr[j] ==0)
                img_ptr[j] = 0;
        }
    }
}

