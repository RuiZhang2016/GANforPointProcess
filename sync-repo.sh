#!/bin/bash

COMMITMSG="auto sync on $HOSTNAME"

EXEC_DIR=$(readlink -f "$(dirname "$0" )" );
CURR_DIR=$(readlink -f "$(pwd)" );
#enter the exec dir (where the repo lies)
cd "$EXEC_DIR"

#detect DVCS
#try git
git status > /dev/null 2> /dev/null
ISGIT=$?

if [ $ISGIT -eq 0 ] ; then
	#bingo, it is git
	DVCSTYPE="git"
	NRMODCOMMAND="git status --porcelain"
	DCVSADD="git add --all ."
	DCVSREMOVE1="git ls-files --deleted"
	DCVSREMOVE2="parallel --gnu -j1 git rm \"{}\""
	DCVSCOMMIT="git commit"
fi

#try mercurial
hg status > /dev/null 2> /dev/null
ISMERCURIAL=$?

if [ $ISMERCURIAL -eq 0 ] ; then
	#must be mercurial
	DVCSTYPE="mercurial"
	NRMODCOMMAND="hg stat"
	DCVSADD="hg add"
	DCVSREMOVE1="hg addremove"	#I need two commands for Git
	DCVSREMOVE2="cat"		#dummy for indirection
	DCVSCOMMIT="hg comm"
fi

#permit setting the commit message by parameter
if [ $# -ge 1 ] ; then
	COMMITMSG="$1"
fi

#starting sync
echo "-> Starting auto sync..."
NRMOD=$( $NRMODCOMMAND | wc -l )
if [ $NRMOD -gt 0 ] ; then
	echo "--> Adding new files, removing old, commiting undeposited changes..."
	#we have modified files not deposited; do them
	#there is a bug of out-of-memory when there are too many files
	$DCVSADD	#add new files
	$DCVSREMOVE1 | $DCVSREMOVE2	#calculate renames
	$DCVSCOMMIT -m "$COMMITMSG"
fi

if [ $DVCSTYPE = "git" ] ; then
	#pull is short for fetch and merge
	git pull ;
	#put back the modifications
	git push ;
else
	#are there incoming?
	hg incoming > /dev/null
	INCOMING=$?
	#are there outgoing?
	hg outgoing > /dev/null
	OUTGOING=$?

	if [ $INCOMING -eq 0 ] ; then
		#we have distant changesets; pull
		hg pull 
		if [ $OUTGOING -gt 0 ] ; then
			#no local changes, just update local repo
			hg up ;
		fi
	fi	

	if [ $OUTGOING -eq 0 ] ; then
		#we have local changes 
		#if distant also, merge and deposit back
		if [ $INCOMING -eq 0 ] ; then
			hg merge 
			hg commit -m "Commit after merge" 
		fi
		hg push 
	fi
fi
	
echo -e "--> Done\n\n\n"
#read -n 1 -p "Press any key to continue..."

#go back to the current dir
cd "$CURR_DIR"
