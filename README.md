# joy-corps
Group repository for http://www.datacombats.com/

## Initial Setup
First, create new folder (\*) - feel free to use any name you like:

```shell
mkdir ~/workspace/dc
```

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
