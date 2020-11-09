#include "slamBase.h"

PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    PointCloud::Ptr cloud ( new PointCloud );

    for (int m = 0; m < depth.rows; m++)
        for (int n=0; n < depth.cols; n++)
        {
            ushort d = depth.ptr<ushort>(m)[n];
            if (d == 0)
                continue;
            PointT p;

            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;
            
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            cloud->points.push_back( p );
        }
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

    return cloud;
}

void computeKeyPointsAndDesp(FRAME & frame)
{
        cv::Ptr<cv::FeatureDetector> detector;
        cv::Ptr<cv::DescriptorExtractor> descriptor;
        
        detector = cv::ORB::create();
        descriptor = cv::ORB::create();

        detector->detect(frame.rgb, frame.kp);
        detector->compute(frame.rgb, frame.kp, frame.desp);
}

RESULT_OF_PNP estimateMotion(FRAME & frame1, FRAME & frame2, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    static ParameterReader pd;
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher; //FLANN 안됨 왠지모르겠음 개빡침.
    matcher.match(frame1.desp, frame2.desp, matches);

    cout<<"find total"<<matches.size()<<" matches."<<endl;
    vector< cv::DMatch> goodMatches;
    double minDis = 9999;
    double good_match_threshold = atof( pd.getData( "good_match_threshold").c_str());
    for(size_t i=0; i<matches.size();i++)
    {
        if(matches[i].distance < minDis)
            minDis = matches[i].distance;
    }
    for(size_t i=0; i<matches.size(); i++)
    {
        if(matches[i].distance< good_match_threshold*minDis)
            goodMatches.push_back(matches[i]);
    }

    cout<<"good matches: "<<goodMatches.size()<<endl;

    vector<cv::Point3f> pts_obj;
    vector<cv::Point2f> pts_img;

    for (size_t i = 0; i < goodMatches.size(); i++)
    {
        cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
        ushort d = frame1.depth.ptr<ushort>(int(p.y))[int(p.x)];
        if (d == 0)
            continue;
        pts_img.push_back(cv::Point2f(frame2.kp[goodMatches[i].trainIdx].pt));

        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt(p.x, p.y, d);
        cv::Point3f pd = point2dTo3d(pt, camera);
        pts_obj.push_back(pd);
    }

    double camera_matrix_data[3][3] = {
        {camera.fx, 0, camera.cx},
        {0, camera.fy, camera.cy},
        {0, 0, 1}};

    cv::Mat cameraMatrix(3, 3, CV_64F, camera_matrix_data);
    cv::Mat rvec, tvec, inliers;

    double min_good_match = atof(pd.getData("min_good_match").c_str());

    if (goodMatches.size() > min_good_match)
    {
        cv::solvePnPRansac(pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 0.99, inliers);
    }

    RESULT_OF_PNP result;
    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;
    result.goodMatches = goodMatches;

    return result;
}

cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    cv::Point3f p;
    p.z = double( point.z ) / camera.scale;
    p.x = ( point.x - camera.cx) * p.z / camera.fx;
    p.y = ( point.y - camera.cy) * p.z / camera.fy;
    return p;
}

Eigen::Isometry3d cvMat2Eigen(cv::Mat &rvec, cv::Mat & tvec)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    Eigen::Matrix3d r;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            r(i, j) = R.at<double>(i, j);

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    T = angle;
    T(0, 3) = tvec.at<double>(0, 0);
    T(1, 3) = tvec.at<double>(1, 0);
    T(2, 3) = tvec.at<double>(2, 0);

    return T;
}

PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    PointCloud::Ptr newCloud = image2PointCloud( newFrame.rgb, newFrame.depth, camera );

    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud( *original, *output, T.matrix() );
    *newCloud += *output;

    static pcl::VoxelGrid<PointT> voxel;
    static ParameterReader pd;
    double gridsize = atof( pd.getData("voxel_grid").c_str() );
    voxel.setLeafSize( gridsize, gridsize, gridsize );
    voxel.setInputCloud( newCloud );
    PointCloud::Ptr tmp( new PointCloud() );
    voxel.filter( *tmp );
    return tmp;
}

FRAME readFrame(RealSense& realsense)
{
    FRAME f;

    realsense.updateFrame();

    realsense.updateColor();
    realsense.updateDepth();
    realsense.updateAlign();
    realsense.drawColor();
    realsense.drawDepth();
    realsense.drawAlign();
    

    f.rgb = realsense.color_mat;
    
    cv::Mat depth = realsense.align_mat.clone();
    depth.convertTo(f.depth, CV_8UC1, 255.0/65536.0);

    f.align = realsense.align_mat.clone();

    // align.convertTo(f.align, CV_8UC1, 255.0/65536.0);
    return f;
}

vector<cv::DMatch> matchingWheel(FRAME & frame1, FRAME & frame2)
{
    static ParameterReader pd;
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher; //FLANN 안됨 왠지모르겠음 개빡침.
    matcher.match(frame1.desp, frame2.desp, matches);

    cout<<"find total"<<matches.size()<<" matches."<<endl;
    vector< cv::DMatch> goodMatches;
    double minDis = 9999;
    double good_match_threshold = atof( pd.getData( "good_match_threshold").c_str());
    double sumDis = 0;
    for(size_t i=0; i<matches.size();i++)
    {
        if(matches[i].distance < minDis)
            minDis = matches[i].distance;
        sumDis += matches[i].distance;

    }
    minDis = sumDis / (matches.size());
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < (good_match_threshold*minDis) + 1)
            goodMatches.push_back( matches[i] );
    }

    cout<<"good matches: "<<goodMatches.size()<<endl;


    double min_good_match = atof(pd.getData("min_good_match").c_str());

    if (goodMatches.size() > min_good_match)
    {
        cout<<"Search complete Wheel!"<<endl;
    }

    return goodMatches;
}