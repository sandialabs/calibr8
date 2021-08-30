NAME=cube
GMODEL_DIR=${CALIBR8DIR}/install/gmodel
GMSH=${CALIBR8DIR}/install/gmsh/bin/gmsh
FROM_GMSH=${CALIBR8DIR}/install/scorec/bin/from_gmsh
SPLIT=${CALIBR8DIR}/install/scorec/bin/split

all: $(NAME)_2p.smb

$(NAME).geo: $(NAME).cpp
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
