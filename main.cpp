
#include <iostream>

using namespace std;
#include <Eigen/Dense>


int main()
{
    Eigen::Matrix3d mat(3,3);
    mat << 1, 2, 3,
        4, 5, 6,
        7, 8, 9;
    cout << mat << endl;
    return 0;
}
