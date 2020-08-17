#include "slamBase.h"

FRAME readFrame( RealSense& realsense, ParameterReader& pd );
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
    char detect_circle_cnt = 0;
    vector<cv::Vec3f> pre_circles;

    while(true)
    {
        cout<<"Reading files "<<currIndex<<endl;
        currFrame = readFrame(realsense);
        FRAME roiFrame, roiFrameGray; // roi이미지 및 roi grayscale 이미지
        cv::Mat binaryimg; // 이진화 이미지
        cv::Mat binaryimg_close;
        cv::Mat edgeimg; // 엣지 이미지
        cv::Mat hsvimg;
        cv::Mat hsvimg_binary;
        cv::Mat hsvimg_binaryblur;
        
        roiFrame.rgb = currFrame.rgb(cv::Rect(start_x,start_y,end_x,end_y));      // roi 이미지 업데이트 -> currframe에서 추출
        roiFrame.depth = currFrame.align(cv::Rect(start_x,start_y,end_x,end_y));  // roidepth 이미지 업데이트 -> currframe에서 추출
        cv::cvtColor(roiFrame.rgb, roiFrameGray.rgb,cv::COLOR_BGR2GRAY);          // roi grayscale img 업데이트 -> roiframe에서 추출
        cv::cvtColor(roiFrame.rgb, hsvimg, cv::COLOR_BGR2HSV);                    // roi hsv img 업데이트 -> roiframe에서 추출

        int lowH = atoi(pd.getData("lowH").c_str());
        int lowS = atoi(pd.getData("lowS").c_str());
        int lowV = atoi(pd.getData("lowV").c_str());
        int highH = atoi(pd.getData("highH").c_str());
        int highS = atoi(pd.getData("highS").c_str());
        int highV = atoi(pd.getData("highV").c_str());
        cv::inRange(hsvimg, cv::Scalar(lowH,lowS,lowV),cv::Scalar(highH,highS,highV), hsvimg_binary);
        cv::blur(hsvimg_binary, hsvimg_binaryblur, cv::Size(3,3));


        int threshold = atoi(pd.getData("threshold").c_str());    // 이진화 threshold value
        cv::threshold(roiFrameGray.rgb, binaryimg, threshold, 255, cv::THRESH_BINARY_INV| cv::THRESH_OTSU); // 이진화
        cv::dilate(binaryimg, binaryimg_close, cv::Mat());
        cv::erode(binaryimg_close, binaryimg_close, cv::Mat());

        int canny_threshold1 = atoi(pd.getData("canny_threshold1").c_str());
        int canny_threshold2 = atoi(pd.getData("canny_threshold2").c_str());
        cv::blur(binaryimg_close, edgeimg, cv::Size(3,3));
        cv::Canny(edgeimg, edgeimg, canny_threshold1, canny_threshold2, 5);

        for(int i = 0; i<edgeimg.rows; i++){
            uchar *img_ptr = edgeimg.ptr<uchar>(i);
            uchar *depth_ptr = roiFrame.depth.ptr<uchar>(i);
            for (int j = 0; j < edgeimg.cols; j++){
                if (((depth_ptr[j + 5] > 3) && (depth_ptr[j - 5] > 3) && ((depth_ptr - 5)[j] > 3) && ((depth_ptr + 5)[j] > 3)))
                    img_ptr[j] = 0;
            }
        }

        int dp = atoi(pd.getData("dp").c_str());
        double minDist = atoi(pd.getData("minDist").c_str());
        int minRadius = atoi(pd.getData("minRadius").c_str());
        int maxRadius = atoi(pd.getData("maxRadius").c_str());
        double accumulate_value = atoi(pd.getData("accumulate_value").c_str());
        int searchcircle = atoi(pd.getData("searchcircle").c_str());
        double w = atoi(pd.getData("low_pass_weight").c_str());
        int detect_threshold = atoi(pd.getData("detect_threshold").c_str());

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


        // data check
        cv::imshow("roi", roiFrame.rgb);
        cv::imshow("roidepth", roiFrame.depth);
        cv::imshow("normal", currFrame.rgb);
        cv::imshow("roiGRAY", roiFrameGray.rgb);
        cv::imshow("edgeimg", edgeimg);
        cv::imshow("binaryimg", binaryimg);
        cv::imshow("binaryimg_close", binaryimg_close);
        cv::imshow("circle", circle_image);
        cv::imshow("hsv", hsvimg);
        cv::imshow("hsv_binary", hsvimg_binary);
        cv::imshow("hsv_binaryblur", hsvimg_binaryblur);



        int check = cv::waitKey(1);
        // cv::imshow("binaryimg", binaryimg);
        if (check == 's')
        {
        }
        else if (check == 'q')
            break;
        currIndex++;
    }

    return 0;
}

