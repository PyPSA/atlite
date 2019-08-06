##########################################
Configuration of Atlite
##########################################

When using Atlite you might want to change
some default locations e.g. for from where
datasets or cutouts are read and to where
new cutouts are saved.


Instead of manually providing configuration of data directories to function calls,
you can create one or more configuration files to hold a standard configuration or
custom configurations for each project.

To create a standard configuration:

* Configuration can be accessed and changed within your code using `atlite.config.<config_variable>`

* To list all `configuration_variables` currently in use `print(atlite.config.ATTRS)`

* Create a new directory `.atlite` in your home directory and place a `config.yaml` file there.
  On unix systems this is `~/.atlite/config.yaml`,
  on windows systems it usually is `C:\\Users\\\<Your Username\>\\.atlite\\config.yaml`.
  
* Copy the settings and format of `atlite/default.config.yaml <atlite/default.config.yaml>`
  and point the directories to where you downloaded or provided the respective data.
  This file is automatically loaded when `import atlite`.
    
* A specific configuration file can be loaded anytime within your code using
  `atlite.config.read(<path to your configuration file>)`

* A specific configuration can be stored anytime from within your code using
  `atlite.config.save(<path to your configuration file>)`
