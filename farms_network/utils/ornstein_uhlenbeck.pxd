# distutils: language = c++

""" OrnsteinUhlenbeck distribution """


from libcpp.random cimport mt19937, normal_distribution


cdef struct OrnsteinUhlenbeckParameters:
    double mu
    double sigma
    double tau
    double dt
    mt19937 random_generator
    normal_distribution[double] distribution


cdef double c_noise_current_update(
    double state, OrnsteinUhlenbeckParameters* params
)
