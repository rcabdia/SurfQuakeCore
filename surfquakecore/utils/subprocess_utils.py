import subprocess as sb



def exc_cmd(cmd, **kwargs):
    """
    Execute a subprocess Popen and catch except.
    :param cmd: The command to execute.
    :param kwargs: All subprocess.Popen(..) kwargs except shell.
    :return: The stdout.
    :except excepts: SubprocessError, CalledProcessError
    :keyword stdin: Default=subprocess.PIPE
    :keyword stdout: Default=subprocess.PIPE
    :keyword stderr: Default=subprocess.PIPE
    :keyword encoding: Default=utf-8
    :keyword timeout: Default=3600
    """
    stdout = kwargs.pop("stdout", sb.PIPE)
    stdin = kwargs.pop("stdin", sb.PIPE)
    stderr = kwargs.pop("stderr", sb.PIPE)
    encoding = kwargs.pop("encoding", "utf-8")
    timeout = kwargs.pop("timeout", 3600)

    # cmd = shlex.split(cmd)
    with sb.Popen(cmd, stdin=stdin, stdout=stdout, stderr=stderr, encoding=encoding, **kwargs) as p:
        try:
            std_out, std_err = p.communicate(timeout=3600)
        except sb.TimeoutExpired:
            p.kill()
            std_out, std_err = p.communicate()  # try again if timeout fails.

        # Print the stdout and stderr to the terminal
        if std_out:
            print(std_out)

        if p.returncode != 0:  # Bad error.
            raise sb.CalledProcessError(p.returncode, std_err)
        elif len(std_err) != 0:
            # Some possible errors trowed by the running subprocess, but not critical.
            print("stderr:", std_err)
            #raise sb.SubprocessError(std_err)
        return std_out


def exc_nll(cmd, **kwargs):
    """
    Execute a subprocess Popen and catch except.
    :param cmd: The command to execute.
    :param kwargs: All subprocess.Popen(..) kwargs except shell.
    :return: The stdout.
    :except excepts: SubprocessError, CalledProcessError
    :keyword stdin: Default=subprocess.PIPE
    :keyword stdout: Default=subprocess.PIPE
    :keyword stderr: Default=subprocess.PIPE
    :keyword encoding: Default=utf-8
    :keyword timeout: Default=3600
    """
    stdout = kwargs.pop("stdout", sb.PIPE)
    stdin = kwargs.pop("stdin", sb.PIPE)
    stderr = kwargs.pop("stderr", sb.PIPE)
    encoding = kwargs.pop("encoding", "utf-8")
    timeout = kwargs.pop("timeout", 3600)

    # cmd = shlex.split(cmd)
    with sb.Popen(cmd, stdin=stdin, stdout=stdout, stderr=stderr, encoding=encoding, **kwargs) as p:
        try:
            std_out, std_err = p.communicate(timeout=3600)
        except sb.TimeoutExpired:
            p.kill()
            std_out, std_err = p.communicate()  # try again if timeout fails.

        # Print the stdout and stderr to the terminal
        print(std_out)

        if p.returncode != 0:  # Bad error.
            raise sb.CalledProcessError(p.returncode, std_err)
        elif len(std_err) != 0:
            # Some possible errors trowed by the running subprocess, but not critical.
            print("stderr:", std_err)
            #raise sb.SubprocessError(std_err)
        return std_out

#
# if __name__ == "__main__":
#
#     command = ['/Users/robertocabiecesdiaz/Documents/SurfQuake/venv/lib/python3.9/site-packages/surfquakecore/binaries/mac_bin/NLL/scat2latlon', "1",
#                './', '/Users/robertocabiecesdiaz/Desktop/all_andorra/nll_all/loc/location.20211001.021028.grid0.loc']
#
#     additional_kwargs = {
#         "stdout": sb.PIPE,
#         "stderr": sb.PIPE,
#         "encoding": "utf-8",
#         "timeout": 10
#     }
#
#     result = exc_cmd(command, **additional_kwargs)
