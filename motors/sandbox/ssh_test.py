import spur

shell = spur.SshShell(hostname="10.66.193.74", username="pi", password="raspberry")
with shell:
    result = shell.run(["python3", "pos.py"])
answer = result.output
ans = answer.decode("utf-8")
ans = float(ans)
print(ans, "test")
print(type(ans))
