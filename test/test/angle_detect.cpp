#include "slamBase.h"

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

    while(1)
    {
        cout<<"Reading files "<<currIndex<<endl;
        currFrame = readFrame(realsense);
        FRAME roiFrame, roiFrameGray;
        roiFrame.rgb = currFrame.rgb(cv::Rect(start_x,start_y,end_x,end_y));
        cv::cvtColor(roiFrame.rgb, roiFrameGray.rgb,cv::COLOR_BGR2GRAY);

        int threshold = atoi(pd.getData("threshold").c_str());
        cv::Mat binaryimg;
        cv::threshold(roiFrameGray.rgb, binaryimg, threshold, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

        cv::Mat edgeimg;
        
        int canny_threshold = atoi(pd.getData("canny_threshold").c_str());
        // cv::blur(binaryimg, edgeimg, cv::Size(3,3));
        cv::blur(roiFrameGray.rgb, edgeimg, cv::Size(3,3));
        cv::Canny(edgeimg, edgeimg, 25, canny_threshold, 3);

        int dp = atoi(pd.getData("dp").c_str());
        double minDist = atoi(pd.getData("minDist").c_str());
        int minRadius = atoi(pd.getData("minRadius").c_str());
        int maxRadius = atoi(pd.getData("maxRadius").c_str());
        double accumulate_value = atoi(pd.getData("accumulate_value").c_str());
        int searchcircle = atoi(pd.getData("searchcircle").c_str());
        cout<<"dp = "<< dp <<endl;
        cout<<"minDist = "<< minDist <<endl;
        cout<<"minRadius = "<< minRadius <<endl;
        cout<<"maxRadius = "<< maxRadius <<endl;
        cout<<"accumulate_vale = "<< accumulate_value <<endl;


        vector<cv::Vec3f> circles;
        cv::HoughCircles(edgeimg, circles, CV_HOUGH_GRADIENT, dp, minDist, searchcircle, accumulate_value, minRadius, maxRadius);
        // cv::HoughCircles(edgeimg, circles, CV_HOUGH_GRADIENT, dp, minDist, edgeimg.rows / 4, accumulate_value, minRadius, maxRadius);
        
        cv::Mat circle_image = roiFrame.rgb.clone();
        if(circles.size()>0){
            for(size_t i = 0;i<circles.size(); i++)
            {
                cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                int radius = cvRound(circles[i][2]);

                cv::circle(circle_image, center, 3, cv::Scalar(0,255,0), -1, 8, 0);
                cv::circle(circle_image, center, radius, cv::Scalar(0,0,255), 3, 8, 0);
            }
        }


        cv::imshow("roi", roiFrame.rgb);
        cv::imshow("normal", currFrame.rgb);
        cv::imshow("roiGRAY", roiFrameGray.rgb);
        cv::imshow("binaryimg", binaryimg);
        cv::imshow("edgeimg", edgeimg);
        cv::imshow("circle", circle_image);



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
