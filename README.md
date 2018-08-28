This respository contains the necessary components to generate, integrate, analyze and visualize neural network models. 

# Installation :

### Local Install

`root_dir> python setup.py build`

`root_dir> export PYTHONPATH=$PYTHONPATH:{path_to_network_generator}`

### Local Pip Install

`root_dir> pip install .`

(You may sudo priviliges for the above!)

# Models

* **Danner Models**:
    * Location : tests/danner_models
    * Description : Neural networks describing mouse spinal circuits

* **Dauns Models**:
    * Location : tests/dauns_models
    * Description : Neural networks describing stick insect spinal circuits
    
# Running the examples

* **Danner Models**

`root_dir> cd tests/danner_models/e_life`

`root_dir/tests/danner_models/e_life> python cpg_danner_network_e_life.py`
