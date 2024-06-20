# Mario AI - 2009

## :gear: Setup (for MacOS) 

### Server

1. Install [rye](https://rye.astral.sh/)
   
    ```sh
    $ curl -sSf https://rye.astral.sh/get | bash
    ```

2. Setup rye environment
    - for `bash`
  
        ```sh
        $ echo 'source "$HOME/.rye/env"' >> ~/.bash_profile
        $ source ~/.bash_profile
        ```

    - for `zsh`
  
        ```sh
        $ echo 'source "$HOME/.rye/env"' >> ~/.zprofile
        $ source ~/.zprofile
        ```

3. Setup python environment
   
    ```sh
    $ rye sync
    ```

4. Activate virtual environment
   
    ```sh
    $ source .venv/bin/activate
    ```

### Client

1. Install JDK (Amazon Corretto 22)
   - Download JDK from the following page
     - https://docs.aws.amazon.com/ja_jp/corretto/latest/corretto-22-ug/downloads-list.html

2. Setup java environment

    ```sh
    $ /usr/libexec/java_home --verbose
    $ export JAVA_HOME=/Library/Java/JavaVirtualMachines/amazon-corretto-22.jdk/Contents/Home
    ```

3. Install Ant for building
   
    ```sh
    $ brew install ant
    ```

## :rocket: How to run

### Server

```sh
$ make run-server
```

### Client

```sh
$ make run-client
```