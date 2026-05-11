extra_sphinx_flag=""

if [ "$1" = "-E" ]; then
    extra_sphinx_flag="-E"
fi

poetry run sphinx-build sphinx/ docs/ -a -j auto ${extra_sphinx_flag}
echo "cellestial.datavil.org" > docs/CNAME
