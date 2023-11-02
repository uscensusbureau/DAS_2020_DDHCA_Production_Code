
.PHONY: assemble package release clean upload

BASE_PACKAGE = ctools
#Default to what's on the path unless set in an env var
PIP3 ?= pip3
PYTHON3 ?= python3

tags:
	etags *.py */*py

assemble: clean
	mkdir -p target/$(BASE_PACKAGE)
	cp -R schema *.py target/$(BASE_PACKAGE)/
	cp *.toml LICENSE README.md target
	mv target/$(BASE_PACKAGE)/setup.py target/
	cp setup.cfg.template target/setup.cfg
	PACKAGES="";\
	for package in $$(grep -v '#' requirements.txt); \
	do PACKAGES="$$PACKAGES    $$package\n"; \
	done; \
	sed -i "s/PACKAGES/$$PACKAGES/g" target/setup.cfg
	cat target/setup.cfg

package: assemble prereqs
	cd target && $(PYTHON3) -m build -n

prereqs:
#Note: We install pkginfo 1.8.3 because 1.9.0 is broken.  When 1.9.1 is available it can be used
	$(PIP3) install --upgrade pip build wheel setuptools pkginfo==1.9.6

install: package 
	$(PIP3) install --force-reinstall target/dist/*.whl


upload: package
	$(PIP3) install --upgrade twine
	$(PYTHON3) -m twine upload --verbose --skip-existing --repository-url ${NEXUS_REPO} target/dist/*

clean:
	rm -rf target

awsome_demo:
	/bin/rm -f tydoc_awsome_demo.html
	$(PYTHON3) tydoc_awsome_demo.py tydoc_awsome_demo.html
	cp tydoc_awsome_demo.html $$HOME/public_html/
