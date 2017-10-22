# joy-corps
Group repository for http://www.datacombats.com/

## Initial Setup
First, create new folder (\*) which we will use for the project - feel free to use any name you like:

```shell
$ mkdir ~/workspace/dc
```

## GitHub 
We host our project on github and therefore we will need to interact with github a lot. But you have to provide your username and password each time you want to push or pull data to and from github. As one of the options to simplify that you could setup ssh keys with github:
https://help.github.com/articles/connecting-to-github-with-ssh/

After that is done you could clone our project into directory we created at (\*)
https://help.github.com/articles/cloning-a-repository/
```shell
$ cd ~/workspace/dc
$ git clone git@github.com:knstntn/joy-corps.git
```

This will create new folder `joy-corps` inside `dc` folder. I prefer to keep things like that to be able to store more project related information like docs outside of the code folder


Git uses a username to associate commits with an identity:
https://help.github.com/articles/setting-your-username-in-git/

GitHub uses the email address set in your local Git configuration to associate commits pushed from the command line with your GitHub account.
https://help.github.com/articles/setting-your-commit-email-address-in-git/

Essentially you would need to do 
```shell
$ git config --global user.name "YOUR_NAME"
$ git config --global user.email "YOUR_EMAIL"
```

All steps described above could be found in https://help.github.com/articles/set-up-git/


## Python Virtual Environment

While working with python it is a common practice to create virtual environments. If you do not have virtual environment tools installed on your machine, please install them using the following instructions: http://virtualenvwrapper.readthedocs.io/en/latest/install.html#basic-installation

After you installed `virtualenv` and related tools we can create our own virtual environment for this project

```shell
mkvirtualenv dc -a ~/workspace/dc/ --python=/usr/bin/python3
```

Here `dc` is the name of the virtual environment, `~/workspace/dc/` is the name of the folder where I would keep my code and `/usr/bin/python3` is the path to `python3`. In your system python3 could be in different place and you can find that out by running `which python3` command in the shell.

Next step is to start working in virtual environment and to do that you should use the following command:

```shell
workon dc
```

This will find virtual environment called `dc` and automatically move you in the corresponding folder.

If you'd like to stop working in the virtual environment you could either close shell or type `deactivate` command in the same shell tab where you've activated the environment.
