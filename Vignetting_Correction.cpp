#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
float calH(float a,float b,float c,Mat GrayImg);
bool check(float a,float b,float c);

int main(int argc, char* argv[])
{
    if(argc<2)
        return 0;
    string path = argv[1];
    Mat img = imread(path,IMREAD_UNCHANGED);
    Mat dst;
    cvtColor(img,img,COLOR_BGR2GRAY); 
    Mat aa = img.clone();   
    // imshow("raw",aa);
    float a=0,b=0,c=0;
    float a_min=0,b_min=0,c_min=0;
    float delta=8;
    float Hmin = calH(a,b,c,img);

    while(delta>1/256)
    {
        float a_temp=a+delta;
        if(check(a_temp,b,c))
        {
            float H = calH(a_temp,b,c,img);
            if(Hmin>H)
            {
                a_min =a_temp;
                b_min =b;
                c_min =c;
                Hmin = H; 
            }
        }
        a_temp=a-delta;
        if(check(a_temp,b,c))
        {
            float H = calH(a_temp,b,c,img);
            if(Hmin>H)
            {
                a_min =a_temp;
                b_min =b;
                c_min =c;
                Hmin = H; 
            }
        }
        float b_temp=b+delta;
        if(check(a,b_temp,c))
        {
            float H = calH(a,b_temp,c,img);
            if(Hmin>H)
            {
                a_min =a;
                b_min =b_temp;
                c_min =c;
                Hmin = H; 
            }
        }
        b_temp=b-delta;
        if(check(a,b_temp,c))
        {
            float H = calH(a,b_temp,c,img);
            if(Hmin>H)
            {
                a_min =a;
                b_min =b_temp;
                c_min =c;
                Hmin = H; 
            }
        }
        float c_temp=c+delta;
        if(check(a,b,c_temp))
        {
            float H = calH(a,b,c_temp,img);
            if(Hmin>H)
            {
                a_min =a;
                b_min =b;
                c_min =c_temp;
                Hmin = H; 
            }
        }
        c_temp=c-delta;
        if(check(a,b,c_temp))
        {
            float H = calH(a,b,c_temp,img);
            if(Hmin>H)
            {
                a_min =a;
                b_min =b;
                c_min =c_temp;
                Hmin = H; 
            }
        }
        delta = delta/2.0;
    }
    // cout<<"***************"<<endl;
    cout<<"amin "<<a_min<<"bmin "<<b_min<<"cmin "<<c_min<<endl;
    Mat result(img.size(),CV_8UC1);
    const int rows = img.rows;
    const int cols = img.cols;  
    float c_x = cols/2.0,c_y = rows/2.0;
    const float d = sqrt(c_x*c_x+c_y*c_y);  

    for(int row=0;row<rows;++row)
    {
        uchar *data = aa.ptr<uchar>(row);
        uchar *value = result.ptr<uchar>(row);
        for(int col=0;col<cols;++col)
        {
            float r=sqrt((row-c_y)*(row-c_y)+(col-c_x)*(col-c_x))/d;
            float r2 = r*r;
            float r4 = r2*r2;
            float r6 = r2*r2*r2;
            float g = 1+a_min*r2+b_min*r4+c_min*r6;
            // this will cause overflow 
            // ToDo: The image should be normalized to the original brightness
            value[col] = round(data[col]*g);
            if(value[col]>255)
            value[col] = 255;
            else if(value[col]<0)
            value[col]=0;
        }
    }
    imshow("HJ",result);
    waitKey(0);
    return 0;
}

bool check(float a,float b,float c)
{
    if ((a>0) && (b==0) && (c==0))
        return true;
    if (a>=0 && b>0 && c==0)
        return true;
    if (c==0 && b<0 && -a<=2*b)
        return true;
    if (c>0 && b*b<3*a*c)
        return true;
    if (c>0 && b*b == 3*a*c && b>=0)
        return true;
    if (c>0 && b*b == 3*a*c && -b>=3*c)
        return true;
    float q_p = (-2*b+sqrt(4*b*b-12*a*c))/(6*c);
    if (c>0 && b*b > 3*a*c && q_p<=0)
        return true;
    float q_d = (-2*b-sqrt(4*b*b-12*a*c))/(6*c);
    if (c>0 && b*b > 3*a*c && q_d>=1)
        return true;
    if (c<0 && b*b > 3*a*c && q_p >=1 && q_d<=0)
        return true;
    return false;
}

float calH(float a,float b,float c,Mat GrayImg)
{
    Mat GrayFloatImg(GrayImg.size(),CV_32FC1);
    const int rows = GrayImg.rows;
    const int cols = GrayImg.cols;    

    float c_x = cols/2.0,c_y = rows/2.0;
    const float d = sqrt(c_x*c_x+c_y*c_y);

    for(int row=0;row<rows;++row)
    {
        uchar *data = GrayImg.ptr<uchar>(row);
        float *value = GrayFloatImg.ptr<float>(row);
        for(int col=0;col<cols;++col)
        {
            float r=sqrt((row-c_y)*(row-c_y)+(col-c_x)*(col-c_x))/d;
            float r2 = r*r;
            float r4 = r2*r2;
            float r6 = r2*r2*r2;
            float g = 1+a*r2+b*r4+c*r6;
            value[col] = data[col]*g;
        }
    }
    // cout<<"GrayFloatImg "<<GrayFloatImg<<endl;

    Mat logImg(GrayImg.size(),CV_32FC1);
    for(int row=0;row<rows;++row)
    {
        float *data = GrayFloatImg.ptr<float>(row);
        float *value = logImg.ptr<float>(row);
        for(int col=0;col<cols;++col)
        {
            value[col] = 255*log(1+data[col])/8;
        }
    }
    // cout<<"logImg "<<logImg<<endl;
    float histogram[256];
    memset(histogram,0,sizeof(float)*256);
    for(int row=0;row<rows;++row)
    {
        float *value = logImg.ptr<float>(row);
        for(int col=0;col<cols;++col)
        {
            int k_d = floor(value[col]);
            int k_u = ceil(value[col]);
            histogram[k_d] += (1+k_d-value[col]);
            histogram[k_u] += (k_u-value[col]);
        }
    }

    float TempHist[256 + 2 * 4];            //    SmoothRadius = 4
    TempHist[0] = histogram[4];                TempHist[1] = histogram[3];    
    TempHist[2] = histogram[2];                TempHist[3] = histogram[1];
    TempHist[260] = histogram[254];            TempHist[261] = histogram[253];
    TempHist[262] = histogram[252];            TempHist[263] = histogram[251];
    memcpy(TempHist + 4, histogram, 256 * sizeof(float));
    //  smooth
    for (int X = 0; X < 256; X++)
        histogram[X] = (TempHist[X] + 2 * TempHist[X + 1] + 3 * TempHist[X + 2] + 4 * TempHist[X + 3] + 5 * TempHist[X + 4] + 4 * TempHist[X + 5] + 3 * TempHist[X + 6] + 2 * TempHist[X + 7]) + TempHist[X + 8] / 25.0f;

    float sum =0;
    for(int i=0;i<256;++i)
    {
        sum += histogram[i];
    }
    float H=0,pk;
    for(int i=0;i<256;++i)
    {
        pk = histogram[i]/sum;
        if(pk!=0)
            H += pk * log(pk); 
    }
    return -H;   
}
