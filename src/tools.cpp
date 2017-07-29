#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */
  
  // Initialise rmse
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // Sanity checks
  if(estimations.size() != ground_truth.size() || estimations.size() == 0){
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  // Accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i){

    VectorXd residual = estimations[i] - ground_truth[i];

    // Element-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  // Calculate root mean squared error
  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;
}

void Tools::CalculateJacobian(const VectorXd& x_state, MatrixXd &Hj) {
  /**
    * Calculate a Jacobian here.
  */
  // State parameters
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  double c1 = px * px + py * py;
  double c2 = sqrt(c1);
  double c3 = (c1 * c2);

  // Compute Jacobian matrix
  Hj << (px / c2), (py / c2), 0, 0,
        -(py / c1), (px / c1), 0, 0,
        py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2, py / c2;
}
