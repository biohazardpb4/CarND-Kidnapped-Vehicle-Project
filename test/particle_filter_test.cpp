#include "gtest/gtest.h"
#include <vector>
#include "../src/helper_functions.h"
#include "../src/particle_filter.h"

using std::vector;

TEST(ParticleFilter, Init) {
  // GPS measurement uncertainty [x [m], y [m], theta [rad]]
  double sigma_pos [3] = {0.3, 0.3, 0.01};
  double sense_x(0), sense_y(0), sense_theta(0);
  
  ParticleFilter pf(10);
  pf.init(sense_x, sense_y, sense_theta, sigma_pos);
  
  EXPECT_EQ(pf.particles.size(), 10);
  for (auto& p : pf.particles) {
    EXPECT_NE(p.x, 0);
    EXPECT_NE(p.y, 0);
    EXPECT_NE(p.theta, 0);
    EXPECT_EQ(p.weight, 1);
  }
}

TEST(ParticleFilter, Prediction) {
  ParticleFilter pf(1);
  // Test without noise to make assertions easier.
  double sigma_pos [3] = {0, 0, 0};
  double x(102), y(65), theta(5.0*M_PI/8.0), delta_t(0.1), velocity(110), yaw_rate(M_PI/8);
  pf.init(x, y, theta, sigma_pos);
  pf.prediction(delta_t, sigma_pos, velocity, yaw_rate);
  
  auto& got = pf.particles[0];
  EXPECT_DOUBLE_EQ(got.x, 97.592046082729098);
  EXPECT_DOUBLE_EQ(got.y, 75.07741997215382);
  EXPECT_DOUBLE_EQ(got.theta, (51.0*M_PI/80.0));
}

TEST(ParticleFilter, UpdateWeights) {
  ParticleFilter pf(1);
  // Test without noise to make assertions easier.
  double sigma_pos [3] = {0, 0, 0};
  double x(4), y(5), theta(-M_PI/2.0);
  pf.init(x, y, theta, sigma_pos);
  
  double sensor_range(-1);
  // Test without noise to make assertions easier.
  double std_landmark [2] = {0, 0};
  
  vector<LandmarkObs> observations;
  LandmarkObs obs1, obs2, obs3;
  obs1.id = 1;
  obs1.x = 2;
  obs1.y = 2;
  observations.push_back(obs1);
  
  obs2.id = 2;
  obs2.x = 3;
  obs2.y = -2;
  observations.push_back(obs2);
  
  obs3.id = 3;
  obs3.x = 0;
  obs3.y = -4;
  observations.push_back(obs3);
  
  Map map_landmarks;
  Map::single_landmark_s l1, l2, l3, l4, l5;
  l1.id_i = 1;
  l1.x_f = 5;
  l1.y_f = 3;
  map_landmarks.landmark_list.push_back(l1);
  
  l2.id_i = 2;
  l2.x_f = 2;
  l2.y_f = 1;
  map_landmarks.landmark_list.push_back(l2);
  
  l3.id_i = 3;
  l3.x_f = 6;
  l3.y_f = 1;
  map_landmarks.landmark_list.push_back(l3);
  
  l4.id_i = 4;
  l4.x_f = 7;
  l4.y_f = 4;
  map_landmarks.landmark_list.push_back(l4);
  
  l5.id_i = 5;
  l5.x_f = 4;
  l5.y_f = 7;
  map_landmarks.landmark_list.push_back(l5);
  
  pf.updateWeights(sensor_range, std_landmark, observations, map_landmarks);
  
  auto& got = pf.particles[0];
  EXPECT_DOUBLE_EQ(got.weight, 4.60e-53);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

