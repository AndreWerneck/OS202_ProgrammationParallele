#include <SFML/Window/Keyboard.hpp>
#include <ios>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <chrono>
#include "cartesian_grid_of_speed.hpp"
#include "vortex.hpp"
#include "cloud_of_points.hpp"
#include "runge_kutta.hpp"
#include "screen.hpp"
#include <mpi.h>

#define ACTION_STEP 'R'
#define ACTION_DOUBLE 'U'
#define ACTION_HALF 'D'
#define ACTION_PLAY 'P'
#define ACTION_STOP 'S'
#define NO_ACTION '_'
#define ACTION_EXIT 'X'

#define TAG_USER_ACTION 0
#define TAG_DATA 1
#define TAG_EXIT 99

// prof's function to read config
auto readConfigFile(std::ifstream &input)
{
    using point = Simulation::Vortices::point;

    int isMobile;
    std::size_t nbVortices;
    Numeric::CartesianGridOfSpeed cartesianGrid;
    Geometry::CloudOfPoints cloudOfPoints;
    constexpr std::size_t maxBuffer = 8192;
    char buffer[maxBuffer];
    std::string sbuffer;
    std::stringstream ibuffer;
    // Lit la première ligne de commentaire :
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lecture de la grille cartésienne
    sbuffer = std::string(buffer, maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    double xleft, ybot, h;
    std::size_t nx, ny;
    ibuffer >> xleft >> ybot >> nx >> ny >> h;
    cartesianGrid = Numeric::CartesianGridOfSpeed({nx, ny}, point{xleft, ybot}, h);
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lit mode de génération des particules
    sbuffer = std::string(buffer, maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    int modeGeneration;
    ibuffer >> modeGeneration;
    if (modeGeneration == 0) // Génération sur toute la grille
    {
        std::size_t nbPoints;
        ibuffer >> nbPoints;
        cloudOfPoints = Geometry::generatePointsIn(nbPoints, {cartesianGrid.getLeftBottomVertex(), cartesianGrid.getRightTopVertex()});
    }
    else
    {
        std::size_t nbPoints;
        double xl, xr, yb, yt;
        ibuffer >> xl >> yb >> xr >> yt >> nbPoints;
        cloudOfPoints = Geometry::generatePointsIn(nbPoints, {point{xl, yb}, point{xr, yt}});
    }
    // Lit le nombre de vortex :
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lit le nombre de vortex
    sbuffer = std::string(buffer, maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    try
    {
        ibuffer >> nbVortices;
    }
    catch (std::ios_base::failure &err)
    {
        std::cout << "Error " << err.what() << " found" << std::endl;
        std::cout << "Read line : " << sbuffer << std::endl;
        throw err;
    }
    Simulation::Vortices vortices(nbVortices, {cartesianGrid.getLeftBottomVertex(),
                                               cartesianGrid.getRightTopVertex()});
    input.getline(buffer, maxBuffer); // Relit un commentaire
    for (std::size_t iVortex = 0; iVortex < nbVortices; ++iVortex)
    {
        input.getline(buffer, maxBuffer);
        double x, y, force;
        std::string sbuffer(buffer, maxBuffer);
        std::stringstream ibuffer(sbuffer);
        ibuffer >> x >> y >> force;
        vortices.setVortex(iVortex, point{x, y}, force);
    }
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lit le mode de déplacement des vortex
    sbuffer = std::string(buffer, maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    ibuffer >> isMobile;
    return std::make_tuple(vortices, isMobile, cartesianGrid, cloudOfPoints);
}

/* Function qui calcule la quantité de points envoyée par chaque processus pour le processus 0
Pour faire la communication en utilisant MPI_Gatherv

Par exemple, le recvcount[2] répresent la quantité de données envoyée par le processus 2
pour le processus 0

Comme on se trouve dans un cas où le processus 0 ne fait que recevoir le message,
le recvcount[0] est evidemment égal à 0
*/
void calcRecvcount(int *recvcounts, int numberOfPoints, int nranks)
{
    recvcounts[0] = 0;
    for (int i = 1; i < nranks; i++)
    {
        int myrank = i - 1;
        int begin = myrank * numberOfPoints / (nranks - 1);
        int end = (myrank + 1) * numberOfPoints / (nranks - 1);

        if (myrank == nranks - 1)
            end = numberOfPoints;

        int sum = end - begin;

        recvcounts[i] = 2 * sum;
    }
}

/* Calc displacements pour la communication MPI_Gatherv*/
void calcDispls(int *displs, int *recvcounts, int numberOfPoints, int nranks)
{
    displs[0] = 0;
    displs[1] = 0;
    for (int i = 2; i < nranks; i++)
    {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
}

int main(int nargs, char *argv[])
{

    // n_ranks: répresent le nombre de processus et
    // my_rank répresent le nombre du processus courant
    int n_ranks, my_rank;

    // initializing MPI
    MPI_Init(&nargs, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // check if has arguments for configuration
    char const *filename;
    if (nargs == 1)
    {
        std::cout << "Usage : vortexsimulator <nom fichier configuration>" << std::endl;
        return EXIT_FAILURE;
    }

    // using configs
    filename = argv[1];
    std::ifstream fich(filename);
    auto config = readConfigFile(fich);
    fich.close();

    std::size_t resx = 800, resy = 600;
    if (nargs > 3)
    {
        resx = std::stoull(argv[2]);
        resy = std::stoull(argv[3]);
    }

    // initializing variables
    auto vortices = std::get<0>(config);
    auto isMobile = std::get<1>(config);
    auto grid = std::get<2>(config);
    auto cloud = std::get<3>(config);

    grid.updateVelocityField(vortices);

    bool animate = false;

    double dt = 0.1; // velocity

    MPI_Status status; // MPI Status variable

    int flag = 0; // Flag to use if there is any message to receive

    /* Ici on divide le processus 0 des autres processus >= 1*/
    /* Le processus 0 est résponsable pour l'affichage d'écran et pour capter les 'inputs' d'user*/
    if (my_rank == 0)
    {
        // screen info
        Graphisme::Screen myScreen({resx, resy}, {grid.getLeftBottomVertex(), grid.getRightTopVertex()});
        std::cout << "######## Vortex simultor ########" << std::endl
                  << std::endl;
        std::cout << "Press P for play animation " << std::endl;
        std::cout << "Press S to stop animation" << std::endl;
        std::cout << "Press right cursor to advance step by step in time" << std::endl;
        std::cout << "Press down cursor to halve the time step" << std::endl;
        std::cout << "Press up cursor to double the time step" << std::endl;

        // initializing variable to hold keyboard's commands
        char action = NO_ACTION;
        auto start = std::chrono::system_clock::now();
        int frames = 0;
        int fps = 0;

        /* Calcule de recvcounts et displacements qui seront utiliser par le processus 0,
            pour recevoir des données en utilisant le MPI_Gatherv
        */
        int cloudRecvcounts[n_ranks];
        int gridRecvcounts[n_ranks];
        int cloudDispls[n_ranks];
        int gridDispls[n_ranks];

        calcRecvcount(cloudRecvcounts, cloud.numberOfPoints(), n_ranks);
        calcRecvcount(gridRecvcounts, grid.numberOfPoints(), n_ranks);
        calcDispls(cloudDispls, cloudRecvcounts, cloud.numberOfPoints(), n_ranks);
        calcDispls(gridDispls, gridRecvcounts, grid.numberOfPoints(), n_ranks);

        auto action_exit_time = std::chrono::system_clock::now();
        float wait_comm_time = n_ranks * 0.05;
        while (myScreen.isOpen())
        {
            /* Vérifie qu'il y a de données a recevoir*/
            MPI_Iprobe(1, TAG_DATA, MPI_COMM_WORLD, &flag, &status);
            if (flag)
            {
                /* Le nombre de vortices n'est pas parallélisé entre les processus,
                seulement le processus 1 est résponsable pour envoyer les données de vortices
                C'est pour ça qu'ici on utilise MPI_Recv au lieu de Gatherv
                */
                MPI_Recv(vortices.data(), vortices.numberOfVortices() * 3, MPI_DOUBLE, 1, TAG_DATA, MPI_COMM_WORLD, &status);

                /*
                Ici, on utilise Gatherv pour recevoir les données qui seront envoyées par les autres processus
                2 Gatherv sont faits, un pour recevoir le cloud data (qui répresente les positions des particules)
                et l'autre pour recevoir le grid data (qui répresent les vectors vitesse de chaque particule)
                */
                MPI_Gatherv(NULL, 0, MPI_DOUBLE, cloud.data(), cloudRecvcounts, cloudDispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(NULL, 0, MPI_DOUBLE, grid.data(), gridRecvcounts, gridDispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                /* Pour chaque fois qu'on reçois des données, on increment les nombres de frames
                Lorsque une seconde se passe, les frames sont affiché dans l'écran et est reseté
                */
                frames++;

                // Code suivant pour mettre a jour l'affichage d'écran avec des nouvelles données
                myScreen.clear(sf::Color::Black);

                std::string strDt = std::string("Time step : ") + std::to_string(dt);
                myScreen.drawText(strDt, Geometry::Point<double>{50, double(myScreen.getGeometry().second - 96)});
                myScreen.displayVelocityField(grid, vortices);
                myScreen.displayParticles(grid, vortices, cloud);

                /*
                    Affichage de FPS dans l'écran
                */
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> diff = end - start;
                if (diff.count() >= 1.0)
                {
                    fps = frames;
                    start = end;
                    frames = 0;
                }
                std::string str_fps = std::string("FPS : ") + std::to_string(fps);
                myScreen.drawText(str_fps, Geometry::Point<double>{300, double(myScreen.getGeometry().second - 96)});
                myScreen.display();
            }

            // on inspecte tous les évènements de la fenêtre qui ont été émis depuis la précédente itération
            sf::Event event;

            /* Ici, on prend l'action d'user et on envoi au process*/
            while (myScreen.pollEvent(event) && action != ACTION_EXIT)
            {
                // event resize screen
                action = NO_ACTION;
                if (event.type == sf::Event::Resized)
                    myScreen.resize(event);
                // event play animation
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::P))
                    action = ACTION_PLAY;
                // event stop animation
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
                    action = ACTION_STOP;
                // event +speed animation
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))
                    action = ACTION_DOUBLE;
                // event -speed animatin
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
                    action = ACTION_HALF;
                // event advance
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
                    action = ACTION_STEP;
                // event close window and terminate other processes
                if (event.type == sf::Event::Closed)
                {
                    action = ACTION_EXIT;
                    action_exit_time = std::chrono::system_clock::now();
                }

                // Broadcast pour tous les autres processus de l'action d'user
                if (action != NO_ACTION)
                    for (int i = 1; i < n_ranks; i++)
                        MPI_Send(&action, 1, MPI_CHAR, i, TAG_USER_ACTION, MPI_COMM_WORLD);
            }

            /*
                Si action est une action de sortie, le processus après avoir envoyé l'action de sortie
                à tous les autres processus, va attendre une certain temps pour recevoir encore les informations manquants,
                pour ne pas risquer d'avoir l'effet de deadlock. Après ce certain temps défini, il sort et finalise le processus
            */
            if (action == ACTION_EXIT)
            {
                std::chrono::duration<double> diff = std::chrono::system_clock::now() - action_exit_time;
                if (diff.count() > wait_comm_time)
                    myScreen.close();
            }
        }
    }

    /* Pour tous les processus différents de 0*/
    else
    {
        char action = NO_ACTION;

        /*
            Les variables ci-dessous sont utilisés pour faire la partie Send du Gatherv
            Les variables cloudPos et gridPos répresent la position dans la mémoire du "buffer"
            de donnés qui seront envoyés par chaque procesus
        */
        auto cloudNumberOfPoints = cloud.numberOfPoints() / (n_ranks - 1);
        auto gridNumberOfPoints = grid.numberOfPoints() / (n_ranks - 1);
        auto cloudPos = (2 * cloudNumberOfPoints) * (my_rank - 1);
        auto gridPos = (2 * gridNumberOfPoints) * (my_rank - 1);

        while (action != ACTION_EXIT)
        {
            bool advance = false;
            // Vérification s'il y a d'actions d'user à recevoir (qui on été envoyés par le processus 0)
            MPI_Iprobe(0, TAG_USER_ACTION, MPI_COMM_WORLD, &flag, &status);
            if (flag)
            {
                // Lecture de l'action d'user en utilisant MPI_Recv
                MPI_Recv(&action, 1, MPI_CHAR, 0, TAG_USER_ACTION, MPI_COMM_WORLD, &status);
                switch (action)
                {
                case ACTION_PLAY:
                    animate = true;
                    break;
                case ACTION_STOP:
                    animate = false;
                    break;
                case ACTION_DOUBLE: // double speed
                    dt *= 2;
                    break;
                case ACTION_HALF: // half speed
                    dt /= 2;
                    break;
                case ACTION_STEP: // advance step
                    advance = true;
                    break;
                case ACTION_EXIT:
                    break;
                default:
                    break;
                }
            }

            if (action != ACTION_EXIT && (animate | advance))
            {
                /*
                    Calcul Runge-Kutta
                */
                if (isMobile)
                    cloud = Numeric::solve_RK4_movable_vortices(dt, grid, vortices, cloud, my_rank - 1, n_ranks - 1);
                else
                    cloud = Numeric::solve_RK4_fixed_vortices(dt, grid, cloud, my_rank - 1, n_ranks - 1);

                auto cloudData = cloud.data();
                auto gridData = grid.data();

                /*
                    Envoie des donnés pour le processus 0
                */
                if (my_rank == 1)
                    MPI_Send(vortices.data(), 3 * vortices.numberOfVortices(), MPI_DOUBLE, 0, TAG_DATA, MPI_COMM_WORLD);
                MPI_Gatherv(&cloudData[cloudPos], 2 * cloudNumberOfPoints, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(&gridData[gridPos], 2 * gridNumberOfPoints, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}