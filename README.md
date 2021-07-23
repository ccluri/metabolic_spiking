# Metabolic spiking

## Installation and requirements
This software was tested on Ubuntu machine (20.04) with Anaconda python package manager. To install the exact python environment, with compatible libraries please use the py36.yml file as follows (you have to edit the last line of this file depending on where anaconda is installed on your machine)

     conda env create -f py36.yml

After this, please activate this specific python environment by

      user@computer:~/metabolic_spiking$ source activate py36
      (py36)user@computer:~/metabolic_spiking$

## Generate figures from the paper

### Figure 1 & supplementary
Please run the steady_state.py first if you want to regenerate everthing from the beginning, you can skip this line if you want to use the pre-generated data.

       (py36)user@computer:~/metabolic_spiking$ python steady_state.py

This will refresh / re-populate the contents of these folders (i) ./reaction_rates/ and (ii) ./steady_state/ with .npz files, after this please type

       (py36)user@computer:~/metabolic_spiking$ python fig1.py
       (py36)user@computer:~/metabolic_spiking$ python fig1_supp.py

You will have Figure1.png and Figure1_supp.png in your folder after this.

### Figure 2 & supplementary
Please run ret_intrinsic_summary.py and fet_intrinsic_summary.py for starting from the beginning, you can skip this if you don't want to use the pre-generated data.

       (py36)user@computer:~/metabolic_spiking$ python ret_intrinsic_summary.py
       (py36)user@computer:~/metabolic_spiking$ python fet_intrinsic_summary.py

This will refresh / re-populate the contents of ./spike_compensation/ folder with some spike_compensate_summary_*.npz files, after this please type

       (py36)user@computer:~/metabolic_spiking$ python fig2.py
       (py36)user@computer:~/metabolic_spiking$ python fig2_supp.py

You will have Figure2.png and Figure2_supp.png in your folder after this. The second line with also add some additional files to the above folder

### Figure 3 dfb neuron
Please run the following

       (py36)user@computer:~/metabolic_spiking$ python fig3_dfb.py

This will output Figure3_dfb.png file

### Figure 3 network model

You have to run the model if you want for other seeds/parameter values than the ones shown in the figure.

       (py36)user@computer:~/metabolic_spiking$ python nw_va.py
       (py36)user@computer:~/metabolic_spiking$ python fig3_nw.py

This will output Figure3_nw.png file

### Figure 3 network model supplementary plot

       (py36)user@computer:~/metabolic_spiking$ python nw_va.py vogels2005
       (py36)user@computer:~/metabolic_spiking$ python nw_va.py metabolic

Each of the above command takes a long time to execute (>12Hrs depending on your machine), and can take up ~10Gb space on your machine. It creates the data for the summary plots from the original paper, and for this paper for one seed and with connectivity of 0.02 (both seed and connectivity were changed, and the results are nearly identical). Only after these two are executed the following command  will work.

     	(py36)user@computer:~/metabolic_spiking$ python fig3_nw_supp.py

When the above is run for the first time, the summary values are calculated and dumped into a .npz file, which is used for the subsequent figure generation calls. In the end, you will have Figure3_nw_supp.png file in your folder.

### Figure 4

Please run the following command

       (py36)user@computer:~/metabolic_spiking$ python fig4.py

This will output Figure4.png


## Other files

* **figure_properties.py**
  Contains the defaults for figure creation
* **utils.py**
  Some useful functions for Recording data during simulations, Q definition is also here
* **channel.py**
  Class definition of an ion channel
* **gates.py**
  Class definitions of ion channel gates and definitions of ROS gates used in PD/MiniSOG/AOX
* **mitochondria.py**
  Class definition of the Nazaret mitochondrial model which is used in this work
* **nazaret_mito.py**
  Original nazaret model converted to python - this file as such is not used in this work
* **avalan_props.py**
  Function definitions used to compute avalanche properties
* **lifcell.py**
  Class def of a hybrid leaky integrate and fire neuron model
* **sncda.py**
  Class def of SNcDA neuron used in figure4 to show PD like symptoms and its metabolic origin


## Contact

Please reach out to me if you find any errors or if you would like to contribute/extend/improve this work.


## License
Copyright 2021 Chaitanya Chintaluri

This software is provided under the GNU General Purpose License version 3.0,
You will find a copy of this license within this folder, or from online here: 
https://www.gnu.org/licenses/gpl-3.0.txt