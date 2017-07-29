#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  VectorXd x = VectorXd(4);

  MatrixXd F = MatrixXd(4, 4);
  F << 1, 0, 1, 0,
       0, 1, 0, 1,
       0, 0, 1, 0,
       0, 0, 0, 1;

  MatrixXd P = MatrixXd(4, 4);
  P << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1000, 0,
       0, 0, 0, 1000;
  
  MatrixXd Q = MatrixXd(4, 4);
  
  ekf_.Init(x, P, F, H_laser_, R_laser_, Q);

  // initialize process noise
  noise_ax = 9;
  noise_ay = 9;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::CalculateFQ(const MeasurementPackage &measurement_pack) {

  // Compute the time elapsed between current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; // Unit: seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  // Modify F matrix to integrate time
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // Set the process covariance matrix Q
  ekf_.Q_ <<  dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
              0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
              dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0,
              0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // First measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      double rho = measurement_pack.raw_measurements_(0);
      double phi = measurement_pack.raw_measurements_(1);
      double rho_dot = measurement_pack.raw_measurements_(2);
      ekf_.x_ << rho * cos(phi), rho * sin(phi), rho_dot * cos(phi), rho_dot * sin(phi);

    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      double px = measurement_pack.raw_measurements_(0);
      double py = measurement_pack.raw_measurements_(1);
      ekf_.x_ << px, py, 0, 0;
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // Done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  // Sanity check
  double dt = measurement_pack.timestamp_ - previous_timestamp_;
  double px = ekf_.x_(0);
  double py = ekf_.x_(1);
  double xy = px * px + py * py;

  if (dt < 0.001 || xy < 0.001) return; // Discard


  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  
  CalculateFQ(measurement_pack);

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

    // Radar updates
    tools.CalculateJacobian(ekf_.x_, Hj_);

    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;

    ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  } else {

    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;

    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // Print output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
