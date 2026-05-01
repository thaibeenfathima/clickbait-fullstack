import compileall
print('compiling src...')
ok = compileall.compile_dir('src', force=True, quiet=1)
print('ok=', ok)
