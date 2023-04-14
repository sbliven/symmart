from importlib_resources import files  # backport for python < 3.9


def get_builtin_image(filename):
    "Get a file pointer to one of the built-in images"
    return files(__package__).joinpath(filename)


def list_builtin_images():
    "List built-in images"
    return [p.name for p in files(__package__).iterdir() if p.name[0] not in "_."]
