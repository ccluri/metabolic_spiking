# Metabolic spiking

## Installation and requirements
This software was tested on Ubuntu machine (20.04) with Anaconda python package manager. To install the exact python environment, with compatible libraries please use the py36.yml file as follows (you have to edit the last line of this file depending on where anaconda is installed on your machine)

     conda env create -f py36.yml

After this, please activate this specific python environment by

      user@computer:~/metabolic_spiking$ source activate py36
      (py36)user@computer:~/metabolic_spiking$

## Generate figures from the paper submitted to PNAS for review
This is very similar to that submitted to bioarxiv but, the PD version is now in supps. So the figure numbering is messed up and will be corrected in the subsequent versions.


|Figure in MS|.odg file /  |.py file|
|------------|-----------|--------|
| Figure 1 | /figure_odgs/Figure1.odg| fig1.py |
| Figure 2 | /figure_odgs/Figure2.odg| fig2.py |
| Figure 3 | /figure_odgs/Figure2.odg| fig2_v6_2.py |
| Figure 4 | /figure_odgs/Figure3.odg| fig3_dfb.py |
| Figure 5 | /figure_odgs/Figure3.odg| fig3_nw.py |
| Figure 6 | /figure_odgs/Figure5_supp.odg| fig_predictions.py |
| Figure S1 |/figure_odgs/Figure1_supp.odg| fig1_supp.py |
| Figure S2 |/figure_odgs/Figure1_supp.odg| fig1_supp_b.py |
| Figure S3 |/figure_odgs/Figure2_supp.odg| fig2_supp.py |
| Figure S4 |/figure_odgs/Figure3_supp.odg| fig3_nw_supp.py |
| Figure S5 | /figure_odgs/Figure4.odg| fig4.py |
| Figure S6 |/figure_odgs/Figure4_supp.odg| -NA- |
| Figure S7 |/figure_odgs/Figure6_supp.odg| -NA- |
| Figure S8 |/figure_odgs/Figure6_supp.odg| -NA- |


### Figures 1, S1 and S2 from the MS
Please run the steady_state.py first if you want to regenerate everthing from the beginning, you can skip this line if you want to use the pre-generated data.

       (py36)user@computer:~/metabolic_spiking$ python steady_state.py

This will refresh / re-populate the contents of these folders (i) ./reaction_rates/ and (ii) ./steady_state/ with .npz files, after this please type

       (py36)user@computer:~/metabolic_spiking$ python fig1.py
       (py36)user@computer:~/metabolic_spiking$ python fig1_supp.py
       (py36)user@computer:~/metabolic_spiking$ python fig1_supp_b.py

You will have Figure1.png, Figure1_supp.png, Figure1_supp_b.png in your folder after this.


For this figure split in two parts type

       (py36)user@computer:~/metabolic_spiking$ python fig1a.py
       (py36)user@computer:~/metabolic_spiking$ python fig1b.py

You will have Figure1a.png and Figure1b.png in your folder after this.

### Figure 2, 3 and S3 from the MS
Please run ret_intrinsic_summary.py and fet_intrinsic_summary.py for starting from the beginning, you can skip this if you don't want to use the pre-generated data.

       (py36)user@computer:~/metabolic_spiking$ python ret_intrinsic_summary.py
       (py36)user@computer:~/metabolic_spiking$ python fet_intrinsic_summary.py

This will refresh / re-populate the contents of ./spike_compensation/ folder with some spike_compensate_summary_*.npz files, after this please type

       (py36)user@computer:~/metabolic_spiking$ python fig2.py
       (py36)user@computer:~/metabolic_spiking$ python fig2_supp.py

You will have Figure2.png and Figure2_supp.png in your folder after this. The second line with also add some additional files to the above folder

### Figure 4 dfb neuron
Please run the following

       (py36)user@computer:~/metabolic_spiking$ python fig3_dfb.py

This will output Figure3_dfb.png file

### Figure 5 network model

You have to run the model if you want for other seeds/parameter values than the ones shown in the figure.

       (py36)user@computer:~/metabolic_spiking$ python nw_va.py
       (py36)user@computer:~/metabolic_spiking$ python fig3_nw.py

This will output Figure3_nw.png file

### Figure 4S network model supplementary plot

       (py36)user@computer:~/metabolic_spiking$ python nw_va.py vogels2005
       (py36)user@computer:~/metabolic_spiking$ python nw_va.py metabolic

Each of the above command takes a long time to execute (>12Hrs depending on your machine), and can take up ~10Gb space on your machine. It creates the data for the summary plots from the original paper, and for this paper for one seed and with connectivity of 0.02 are reported. Please note that both seed and connectivity were also tested and the results are nearly identical (not shown). This is easily achivied in the code). Only after these two are executed the following command  will work. Many files are created (./netsim_results/20/*_spks.pkl) with actual spike times and other useful variables.  

After this, we compute the avalanche properties for the datasets from above. For this use the following for bulk summary analysis.

     	(py36)user@computer:~/metabolic_spiking$ python avalan_props.py vogels2005
	(py36)user@computer:~/metabolic_spiking$ python avalan_props.py metabolic

These commands will produce as many files with the summary data as pickles (./netsim_results/20/*_summary.pkl)

     	(py36)user@computer:~/metabolic_spiking$ python fig3_nw_supp.py

When the above is run for the first time, the summary values are calculated and dumped into a .npz file, which is used for the subsequent figure generation calls. In the end, you will have Figure3_nw_supp.png file in your folder.

### Figure S5 (PD figure)

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