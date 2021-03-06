#include "slamBase.h"
#include <algorithm>
#define CV_RETR_TREE 3
#define CV_CHAIN_APPROX_SIMPLE 2
#define CV_HOUGH_GRADIENT 3

void depth_delete(cv::Mat &edgeimg, cv::Mat &depth);

void get_point_cloud(FRAME frame, RealSense realsense)
{

}

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

    int contour_size = atoi(pd.getData("contour_size").c_str());
    double contours_epsilon = atoi(pd.getData("contours_epsilon").c_str());

    char detect_circle_cnt = 0;
    vector<cv::Vec3f> pre_circles;

    cv::Mat elements(7,7,CV_8U, cv::Scalar(1));
    while(true)
    {
        // cout<<"Reading files "<<currIndex<<endl;
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
        cv::Mat addEdgeErode;
        cv::Mat addEdge;
        
        roiFrame.rgb = currFrame.rgb(cv::Rect(start_x,start_y,end_x,end_y));      // roi 이미지 업데이트 -> currframe에서 추출
        roiFrame.depth = currFrame.depth(cv::Rect(start_x,start_y,end_x,end_y));  // roidepth 이미지 업데이트 -> currframe에서 추출
        cv::cvtColor(roiFrame.rgb, roiFrameGray.rgb,cv::COLOR_BGR2GRAY);          // roi grayscale img 업데이트 -> roiframe에서 추출
        cv::cvtColor(roiFrame.rgb, hsvimg, cv::COLOR_BGR2HSV);                    // roi hsv img 업데이트 -> roiframe에서 추출
        
        cv::inRange(hsvimg, cv::Scalar(lowH,lowS,lowV),cv::Scalar(highH,highS,highV), hsvimg_binary);
        cv::dilate(hsvimg_binary, hsvimg_binaryMopol, elements);
        cv::erode(hsvimg_binaryMopol, hsvimg_binaryMopol, elements);

        cv::threshold(roiFrameGray.rgb, binaryimg, threshold, 255, cv::THRESH_BINARY_INV| cv::THRESH_OTSU); // 이진화
        cv::dilate(binaryimg, binaryimg_close, elements);
        cv::erode(binaryimg_close, binaryimg_close, elements);

        cv::blur(binaryimg_close, edgeimg_close, cv::Size(3,3));  // 닫힘연산한 일반이미지 블러링
        cv::Canny(edgeimg_close, edgeimg_close, canny_threshold1, canny_threshold2, 5);  //  일반 이미지 닫힘연산 캐니엣지
        // depth_delete(edgeimg_close, roiFrame.depth);
        // cv::imshow("edgeimg_close", edgeimg_close);  // RGB edge


        // cv::imshow("hsv_Mopol_pre", hsvimg_binaryMopol); // HSV
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
        cv::Canny(hsvimg_binaryblur, edgeimg, canny_threshold1, canny_threshold2, 5); // hsv 모폴로지 이진 이미지를 캐니 에지로 연산

        //contours를 이용한 긴에지 검출
        vector<vector<cv::Point>> contours;
        vector<vector<cv::Point>> contours_correct;
        vector<vector<cv::Point>> contours_Approximate;
        vector<cv::Vec4i> hierarchy;
        vector<cv::RotatedRect> ellipse;

        cv::Mat contoursRGB = roiFrame.rgb.clone();
        cv::findContours(edgeimg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);  // edge에서 contours 추출
        // cv::findContours(hsvimg_binaryMopol, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE); // hsv mopology image에서 contours 추출
        cv::Mat drawing_correct = cv::Mat::zeros(roiFrame.rgb.size(), CV_8UC3);
        cv::Mat drawing_approximate = cv::Mat::zeros(roiFrame.rgb.size(), CV_8UC3);

        cv::Scalar color = cv::Scalar(0, 0, 255);
        for(int i = 0; i<contours.size(); i++){
            if(contours[i].size() >= contour_size)
                contours_correct.push_back(contours[i]); // edge correcting 
        }
        for(int i = 0; i<contours_correct.size(); i++){
            vector<cv::Point> tmp;
            cv::approxPolyDP(contours_correct[i], tmp, contours_epsilon, true);
            contours_Approximate.push_back(tmp);
        }
        
        //contours drawing
        for(int i = 0; i<contours_correct.size(); i++){
            cv::drawContours(drawing_correct, contours_correct, i, color, 1, 16, hierarchy, 0, cv::Point());
        }

        for (int i = 0; i < contours_Approximate.size(); i++){
            cv::drawContours(drawing_approximate, contours_Approximate, i, color, 1, 16, hierarchy, 0, cv::Point());
        }

        for(int i = 0; i<contours_correct.size();i++){
            if(contours_correct[i].size() >= 5)
                cv::RotatedRect tmp = cv::fitEllipse(contours_correct[i]);
        }
        
        // cv::imshow("roiFrame", roiFrame.rgb);
        cv::imshow("contours_correct", drawing_correct);
        // cv::imshow("contours_Approximate", drawing_approximate);
        cv::Mat tmp_countours;
        cv::Mat circle_image = roiFrame.rgb.clone();
        vector<cv::Vec3f> circles;
        if (contours_Approximate.size() > 0){
            cv::cvtColor(drawing_approximate, tmp_countours, cv::COLOR_BGR2GRAY);                                 // roi grayscale img 업데이트 -> roiframe에서 추출
            cv::threshold(tmp_countours, tmp_countours, threshold, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU); // 이진화
            cv::Canny(tmp_countours, tmp_countours, canny_threshold1, canny_threshold2, 5);                       // hsv 모폴로지 이진 이미지를 캐니 에지로 연산
            cv::imshow("contoursEdge_Approximate", tmp_countours);
            cv::HoughCircles(tmp_countours, circles, CV_HOUGH_GRADIENT, dp, minDist, searchcircle, accumulate_value, minRadius, maxRadius);
            // cout << "cicles.size = " << circles.size() << endl;

            // 가중평균필터
            if (circles.size() > 0)
            {
                if (detect_circle_cnt < detect_threshold)
                {
                    detect_circle_cnt++;
                    pre_circles = circles;
                }
                else{
                    circles[0][0] = ((w * circles[0][0]) / 10) + ((10 - w) * pre_circles[0][0] / 10);
                    circles[0][1] = ((w * circles[0][1]) / 10) + ((10 - w) * pre_circles[0][1] / 10);
                    circles[0][2] = ((w * circles[0][2]) / 10) + ((10 - w) * pre_circles[0][2] / 10);
                    for (size_t i = 0; i < circles.size(); i++){
                        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                        int radius = cvRound(circles[i][2]);

                        cv::circle(circle_image, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
                        cv::circle(circle_image, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
                        
                        cv::Point left_vertex(cvRound(circles[i][0])-cvRound(circles[i][2])+7,cvRound(circles[i][1]));
                        cv::Point right_vertex(cvRound(circles[i][0])+cvRound(circles[i][2])-7,cvRound(circles[i][1]));
                        cv::circle(circle_image, left_vertex, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
                        cv::circle(circle_image, right_vertex, 3, cv::Scalar(255, 0, 0), -1, 8, 0);
                        
                        ///object length calculator
                        ushort ld = currFrame.align.ptr<ushort>(left_vertex.y)[left_vertex.x+170];
                        double lz = (double)ld/ 1000;
                        double lx = ((double)left_vertex.x + 170 - 318.653) * lz / 384.291; 
                        double ly = ((double)left_vertex.y - 249.416) * lz / 383.923;
                        // cout << "lx: " << lx << " ly: " << ly <<" lz: " << lz <<endl;

                        ushort rd = currFrame.align.ptr<ushort>(right_vertex.y)[right_vertex.x+170];
                        double rz = (double)rd / 1000;
                        double rx = ((double)right_vertex.x + 170 - 318.653) * rz / 384.291; 
                        double ry = ((double)right_vertex.y - 249.416) * rz / 383.923;
                        // cout << "rx: " << rx << " ry: " << ry <<" rz: " << rz <<endl;
                        cout << "x:" << circles[0][0] << ", y:" << circles[0][1] << ", radius:" << circles[0][2] << " real : ";
                        cout << sqrt(pow((rx-lx),2)+pow((ry-ly),2)+pow((rz-lz),2))*1000<<endl;
                    }
                    pre_circles = circles;
                }

            }else detect_circle_cnt = 0;
        }

            // // data check
            // cv::imshow("binaryimg_close", binaryimg_close); // RGB
            // cv::imshow("hsv", hsvimg);                      // HSV
            // // cv::imshow("hsv_binary", hsvimg_binary);
            // cv::imshow("hsv_Mopol", hsvimg_binaryMopol);     // AND 연산
            // cv::imshow("hsv_binaryblur", hsvimg_binaryblur); // 위에 블러
            // cv::imshow("hsvedge", hsvedge);
            // cv::imshow("tmp_countours", tmp_countours);     // 엣지
            cv::imshow("circle", circle_image); //원

            // cv::imshow("currFrame.align", currFrame.align);
            // cv::imshow("currFrame.depth", currFrame.depth);
            // cv::imshow("currFrame.rgb", currFrame.rgb);
            
            // // cv::imshow("roiFrame.align", roiFrame.align);
            // cv::imshow("roiFrame.depth", roiFrame.depth);
            cv::imshow("roiFrame.rgb", roiFrame.rgb);
            
            // cout << currFrame.align.cols <<" "<< currFrame.align.rows<<endl;

            int check = cv::waitKey(1);
            if (check == 's'){}
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
            if ((depth_ptr[j] > 5)|| depth_ptr[j] ==0)
                img_ptr[j] = 0;
        }
    }
}