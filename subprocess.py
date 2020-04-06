import subprocess

# verbose = False
git_clone_proc_rc = 0
cmd = ['git', 'clone', 'https://example.com/git/project.git']
if verbose:
    git_clone_proc_rc = subprocess.call(cmd)
else:
    git_clone_proc_rc = subprocess.call(cmd, \
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if git_clone_proc_rc != 0:
    print("error: Cannot clone repository from {0}".format(cloneURL));
sys.exit(1);


# other example
passwd_file = "input.txt"
priv_key_filename = "id_rsa"
passwd = subprocess.check_output(\
        ['openssl', 'aes-256-cbc', '-d', '-salt', '-in', passwd_file, \
         '-pass', 'file:'+priv_key_filename]).decode('utf-8').rstrip();


passwd = "mysecrect"
p1 = subprocess.Popen(['echo', passwd], stdout=subprocess.PIPE);
p2rc = subprocess.call(\
        ['openssl', 'aes-256-cbc', '-salt', '-out', passwd_file, '-pass', 'file:'+priv_key_filename], \
        stdin=p1.stdout);
p1.wait();
if p2rc != 0:
    print("error: Could not create encrypted file {}.".format(passwd_file));
    sys.exit(1);
