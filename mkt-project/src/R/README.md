### Light Documentation

Here, we provide some light documentation on the various functions and S3 Classes

#### Data Reader

Our data assets are managed via lightweight manifest descriptions that
are in the JSON format.

* `DataReader.R`: defines a class that exposes a
  lightweight manager for manifest objects. Given the source of the JSON file, a representative   object is returned which can be queried for data.
  
* `Rule.R`: defines a class that is used to create Trading Rules that can act as features.

* `Indicators.R`: helper class to generate various indicators that can act as features.