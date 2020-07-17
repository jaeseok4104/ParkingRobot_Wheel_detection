#include "slamBase.h"

int main(void)
{
    RealSense realsense;
    ParameterReader pd;
    cout<<"Realsense loading complete!!"<<endl;
    FRAME matchimg, currFrame;
    matchimg.rgb = cv::imread("/home/park/Work/parkingrobo/test/dataset/wheel.jpg") ;

    int currIndex = 0;

    // CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();


    int start_x = atoi(pd.getData("start_x").c_str());
    int start_y = atoi(pd.getData("start_y").c_str());
    int end_x = atoi(pd.getData("end_x").c_str());
    int end_y = atoi(pd.getData("end_y").c_str());

        computeKeyPointsAndDesp(matchimg);

        cout << "wheel featcher size = "<<matchimg.kp.size()<<endl;

        cv::Mat imgShow;
        cv::drawKeypoints(matchimg.rgb, matchimg.kp, imgShow, cv::Scalar::all(-1));
    while(1)
    {
        cout<<"Reading files "<<currIndex<<endl;
        currFrame = readFrame(realsense);
        FRAME roiFrame, roiFrameGray;
        roiFrame.rgb = currFrame.rgb(cv::Rect(start_x,start_y,end_x,end_y));
        cv::cvtColor(roiFrame.rgb, roiFrameGray.rgb,cv::COLOR_BGR2GRAY);
        computeKeyPointsAndDesp(roiFrame);
        cv::Mat roikeydis;
        cv::drawKeypoints(roiFrame.rgb, roiFrame.kp, roikeydis, cv::Scalar::all(-1));
        
        vector<cv::DMatch> goodMatches;
        goodMatches = matchingWheel(roiFrame, matchimg);

        cv::imshow("roi", roiFrame.rgb);
        cv::imshow("normal", currFrame.rgb);
        cv::imshow("keypoints", imgShow);
        cv::imshow("roiGRAY", roiFrameGray.rgb);

        // cv::imshow("currKeyPoint", roikeydis);
        // cv::Mat imgMatches;
        // cv::drawMatches(roiFrame.rgb, roiFrame.kp, matchimg.rgb, matchimg.kp, goodMatches, imgMatches);
        // cv::imshow("good matches", imgMatches);
        int check = cv::waitKey(1);
        if (check == 's')
        {
        }        else if (check == 'q')
            break;
        currIndex++;
    }

    return 0;
}