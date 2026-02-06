rm -r build *.egg-info

export VERSIONS=(9 10 11 12 13 14)

for ver in $VERSIONS; do
	echo "$ver"
	rm -r "~/.virtualenvs/zmesh3$ver"
	mkvirtualenv -p "python3.$ver" "zmesh3$ver"
	uv pip install -r requirements.txt -r dev_requirements.txt
	uv pip install pbr setuptools wheel build twine
done

workon zmesh39
python setup.py sdist

for plat in arm64 x86_64; do
	export ARCHFLAGS="-arch $plat"
	if [ "$plat" = "arm64" ]; then
		export PLATNAME=macosx_11_0_arm64
	else
		export PLATNAME=macosx_10_9_x86_64
	fi

	for ver in $VERSIONS; do
		workon "zmesh3$ver"
		# if [ "$plat" = "arm64" ]; then
		# 	python setup.py develop
		# 	python -m pytest -v automated_test.py
		# fi
		python -m build --wheel --config-setting=--build-option="--plat-name=$PLATNAME"
	done
done
