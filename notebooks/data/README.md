PV Solar Farm Data
---
Data are ordered, timestamped, single-valued metrics.
  - solar_edge_tageswerte_prepared.csv: Original data set representin 5 years of daily energy production of PV Solar Farm
  - train_data_energy.csv: subset of the original data used for training purposes
  - eval_data_energy.csv: subset of the original data user for evaluation purposes
  - Labeled_SolarEdge_Eval_Dataset-original.csv: the original data set labeled by the experts containing the anomalies

Synthetical Solar Farm Data
---
Data is derived from the original data set
  - synthetic_data_anomalies_energy.csv: labeled original data set by domain experts

NAB Data Corpus
---

Data are ordered, timestamped, single-valued metrics. All data files contain anomalies, unless otherwise noted.


### Real data

	- machine_temperature_system_failure.csv: Temperature sensor data of an
	internal component of a large, industrial mahcine. The first anomaly is a
	planned shutdown of the machine. The second anomaly is difficult to detect and
	directly led to the third anomaly, a catastrophic failure of the machine.
  - train_data_not_cleaned_with_anomaly.csv: Clone of the original data set
  - train_data.csv: original data stripped of the defined anomalies for model to learn normal behaviour
  - data_anomalies_machine.csv: original data with labeled anomalies



