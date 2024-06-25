NAME = notch2D
GMODEL_DIR=${CALIBR8DIR}/install/gmodel
GMSH=${CALIBR8DIR}/install/gmsh/bin/gmsh
FROM_GMSH=${CALIBR8DIR}/install/scorec/bin/from_gmsh
SPLIT=${CALIBR8DIR}/install/scorec/bin/split

GMSH=/Applications/Gmsh.app/Contents/MacOS/gmsh
FROM_GMSH=/Users/bngranz/codes/calibr8-project/scorec/install/bin/from_gmsh
SPLIT=/Users/bngranz/codes/calibr8-project/scorec/install/bin/split
GMODEL_DIR=/Users/bngranz/codes/calibr8-project/gmodel/install

all: $(NAME)_2p.smb

$(NAME).geo:
	mpicxx -std=c++11 $(NAME).cpp -o $(NAME) -I${GMODEL_DIR}/include -L${GMODEL_DIR}/lib -lgmodel
	./$(NAME)

$(NAME).msh: $(NAME).geo
	${GMSH} -3 $(NAME).geo -o $(NAME).msh -format msh2

$(NAME).smb: $(NAME).msh
	${FROM_GMSH} $(NAME).dmg $(NAME).msh $(NAME).smb

$(NAME)_2p.smb: $(NAME).smb
	mpirun -n 2 ${SPLIT} $(NAME).dmg $(NAME).smb $(NAME)_2p.smb 2

clean:
	rm -rf $(NAME) $(NAME).geo $(NAME).dmg $(NAME).msh $(NAME)*.smb out*
