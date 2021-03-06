/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
#include "multiv_gauss.h"

using std::normal_distribution;
using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  normal_distribution<double> dist_x(x, std[0]),
    dist_y(y, std[1]), dist_theta(theta, std[2]);
  for (int i = 0; i < num_particles; i++) {
    // Initialize all particles to first position (based on estimates of x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    Particle p;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  const auto& v(velocity);
	for (auto& p : particles) {
      if (yaw_rate != 0.0) {
      	p.x = p.x + v/yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      	p.y = p.y + v/yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
      } else {
        p.x += v*cos(p.theta);
        p.y += v*sin(p.theta);
      }
      p.theta = p.theta + yaw_rate*delta_t;
      
      // Add noise
      normal_distribution<double> dist_x(p.x, std_pos[0]), dist_y(p.y, std_pos[1]), dist_theta(p.theta, std_pos[2]);
      p.x = dist_x(gen);
      p.y = dist_y(gen);
      p.theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(const Map &map_landmarks, /*vector<LandmarkObs> predicted,*/ 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int oi = 0; oi < observations.size(); oi++) {
    auto& o = observations[oi];
    int min_id;
    double min_d;
    for (int i = 0; i < map_landmarks.landmark_list.size(); i++) {
      auto& p = map_landmarks.landmark_list[i];
      auto d = dist(o.x, o.y, p.x_f, p.y_f);
      // std::cout << "observation index " << oi << " -- got distance " << d << " for landmark " << p.id_i << std::endl;
      if (d < min_d || i == 0) {
        // std::cout << "keeping as min for observation" << std::endl;
        min_id = p.id_i;
        min_d = d;
      }
    }
    o.id = min_id;
    // learn now to store references in vectors
    observations[oi] = o;
    // std::cout << "observation index " << oi << " associated with landmark id " << o.id << std::endl;
  }
}

//void ParticleFilter::carObsToMapCoords(const vector<LandmarkObs> &observations

  /**
   * updateWeights Updates the weights for each particle based on the likelihood
   *   of the observed measurements. 
   * @param sensor_range Range [m] of sensor
   * @param std_landmark[] Array of dimension 2
   *   [Landmark measurement uncertainty [x [m], y [m]]]
   * @param observations Vector of landmark observations
   * @param map Map class containing map landmarks
   */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */  
  double total_weight=0;
  for (int parti = 0; parti < particles.size(); parti++) {
    auto& p = particles[parti];
    vector<LandmarkObs> t_observations;
    
    bool printed_nan_bug = false;
    
    for (auto& o : observations) {
      // convert observation into map space
      LandmarkObs t_observation;
      t_observation.x = p.x + (cos(p.theta)*o.x) - (sin(p.theta)*o.y);
      t_observation.y = p.y + (sin(p.theta)*o.x) + (cos(p.theta)*o.y);
      t_observation.id = o.id;
      t_observations.push_back(t_observation);
      
      if (std::isnan(t_observation.x) || std::isnan(t_observation.y)) {
        if (!printed_nan_bug) {
          printed_nan_bug = true;
          std::cout << "particle index " << parti << " p.x " << p.x << " p.y " << p.y << " t_observation.x " << t_observation.x << " t_observation.y " << t_observation.y << " p.theta " << p.theta << std::endl;
          exit(1);
        }
      }
    }
    // find the closest landmark to each observation
    dataAssociation(map_landmarks, t_observations);
    // calculate prob of each landmark/observation pair and accumulate via multiplication
    double total_prob(1.0);
    
    // Tracking associations for debugging
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    
    // only t_observations has updated IDs. "observations" does not since it wasn't modified.
    for (auto& o : t_observations) {
      const auto& std_x = std_landmark[0];
      const auto& std_y = std_landmark[1];
      // TODO: clean this up
      auto landmark_x = map_landmarks.landmark_list[0].x_f;
      auto landmark_y = map_landmarks.landmark_list[0].y_f;
      for (auto& lm : map_landmarks.landmark_list) {
        if (lm.id_i == o.id) {
          landmark_x = lm.x_f;
          landmark_y = lm.y_f;
        }
      }
      associations.push_back(o.id);
      sense_x.push_back(landmark_x);
      sense_y.push_back(landmark_y);
      
      total_prob *= multiv_prob(std_x, std_y, o.x, o.y, landmark_x, landmark_y);
    }
    SetAssociations(p, associations, sense_x, sense_y);
    // Assign final weight (accumulated probability)
    p.weight = total_prob;
    //std::cout << "particle index " << parti << " pre-normalized weight " << p.weight << std::endl;
    // getting around vector reference stuff
    particles[parti] = p;
    total_weight += total_prob;
  }
  // std::cout << "number of observations " << observations.size() << " total weight " << total_weight << std::endl;
  // Normalize weights
  for (auto& p : particles) {
    p.weight /= total_weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // std::cout << "before max prob" << std::endl;
  double max_prob = particles[0].weight;
  for (auto& p : particles) {
    if (p.weight > max_prob) {
      max_prob = p.weight;
    }
  }
  
  // std::cout << "before sampling" << std::endl;
  std::vector<Particle> sampled_particles;
  for (auto& p : particles) {
    int index = rand()%particles.size();
    double beta = max_prob*2;
    while (particles[index].weight < beta) {
      beta -= particles[index].weight;
      index = (index+1)%particles.size();
    }
    sampled_particles.push_back(particles[index]);
  }
  particles = sampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}