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

void calcRecvcount(int *recvcounts, int numberOfPoints, int nranks)
{
    recvcounts[0] = 0;
    int sum = 2 * numberOfPoints / (nranks - 1);
    for (int i = 1; i < nranks; i++)
    {
        recvcounts[i] = sum;
    }
}

void calcDispls(int *displs, int numberOfPoints, int nranks)
{
    displs[0] = 0;
    displs[1] = 0;
    int sum = 2 * numberOfPoints / (nranks - 1);
    for (int i = 2; i < nranks; i++)
    {
        displs[i] = displs[i - 1] + sum;
    }
}

int main(int nargs, char *argv[])
{

    // number of process and process number
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

    bool animate = true;

    double dt = 0.1; // velocity

    MPI_Status status; // mpi_status for MPI_Iprobe
    MPI_Request request;
    int flag = 0;           // flag to use if there is any message to receive
    int terminate_flag = 0; // flag to use only if want to terminate processes

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
        int countExited = 0;

        int cloudRecvcounts[n_ranks];
        int gridRecvcounts[n_ranks];
        int cloudDispls[n_ranks];
        int gridDispls[n_ranks];

        calcRecvcount(cloudRecvcounts, cloud.numberOfPoints(), n_ranks);
        calcRecvcount(gridRecvcounts, grid.numberOfPoints(), n_ranks);
        calcDispls(cloudDispls, cloud.numberOfPoints(), n_ranks);
        calcDispls(gridDispls, grid.numberOfPoints(), n_ranks);

        auto timeoutFinalize = std::chrono::system_clock::now();
        while (myScreen.isOpen())
        {
            MPI_Iprobe(1, TAG_DATA, MPI_COMM_WORLD, &flag, &status);
            if (flag)
            {
                MPI_Recv(vortices.data(), vortices.numberOfVortices() * 3, MPI_DOUBLE, 1, TAG_DATA, MPI_COMM_WORLD, &status);

                MPI_Gatherv(NULL, 0, MPI_DOUBLE, cloud.data(), cloudRecvcounts, cloudDispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                MPI_Gatherv(NULL, 0, MPI_DOUBLE, grid.data(), gridRecvcounts, gridDispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }

            if (flag)
            {
                frames++;

                // updating screen
                myScreen.clear(sf::Color::Black);

                std::string strDt = std::string("Time step : ") + std::to_string(dt);
                myScreen.drawText(strDt, Geometry::Point<double>{50, double(myScreen.getGeometry().second - 96)});
                myScreen.displayVelocityField(grid, vortices);
                myScreen.displayParticles(grid, vortices, cloud);

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
            // while inspecting screen and the command is not T (terminate other processes) or X (terminate this process)
            while (myScreen.pollEvent(event) && action != ACTION_EXIT)
            {
                // event resize screen
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
                    action = ACTION_EXIT;
                // if a key is pressed, send this key to process 1
                if (action != NO_ACTION)
                {
                    for (int i = 1; i < n_ranks; i++)
                        MPI_Send(&action, 1, MPI_CHAR, i, TAG_USER_ACTION, MPI_COMM_WORLD);
                }
                if (action == ACTION_EXIT)
                    timeoutFinalize = std::chrono::system_clock::now();
            }

            auto now = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = now - timeoutFinalize;
            if (action == ACTION_EXIT && diff.count() > (0.05 * n_ranks))
            {
                myScreen.close();
            }
        }
    }
    else
    {
        char action = NO_ACTION;
        while (action != ACTION_EXIT)
        {
            bool advance = false;
            // checking if process 0 has sent messages
            MPI_Iprobe(0, TAG_USER_ACTION, MPI_COMM_WORLD, &flag, &status);
            if (flag)
            {
                // reading keyboard pressed
                MPI_Recv(&action, 1, MPI_CHAR, 0, TAG_USER_ACTION, MPI_COMM_WORLD, &status);
                switch (action)
                {
                case ACTION_EXIT:
                    break;
                case ACTION_PLAY: // play
                    animate = true;
                    break;
                case ACTION_STOP: // stop
                    animate = false;
                    break;
                case ACTION_DOUBLE: //+speed
                    dt *= 2;
                    break;
                case ACTION_HALF: //-speed
                    dt /= 2;
                    break;
                case ACTION_STEP: // advance
                    advance = true;
                    break;
                default:
                    break;
                }
            }

            // if not stopped, calculate
            if (action != ACTION_EXIT && (animate | advance))
            {
                if (isMobile)
                {
                    cloud = Numeric::solve_RK4_movable_vortices(dt, grid, vortices, cloud, my_rank - 1, n_ranks - 1);
                }
                else
                {
                    cloud = Numeric::solve_RK4_fixed_vortices(dt, grid, cloud, my_rank - 1, n_ranks - 1);
                }

                // sending data to 0 (affichage)
                // multiply *2 because Geometry::Vector<double> espected in grid is 2D
                if (my_rank == 1)
                    MPI_Send(vortices.data(), vortices.numberOfVortices() * 3, MPI_DOUBLE, 0, TAG_DATA, MPI_COMM_WORLD);

                auto cloudData = cloud.data();
                auto cloudNumberOfPoints = cloud.numberOfPoints() / (n_ranks - 1);
                auto cloudPos = (2 * cloudNumberOfPoints) * (my_rank - 1);
                MPI_Gatherv(&cloudData[cloudPos], 2 * cloudNumberOfPoints, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                auto gridData = grid.data();
                auto gridNumberOfPoints = grid.numberOfPoints() / (n_ranks - 1);
                auto gridPos = (2 * gridNumberOfPoints) * (my_rank - 1);
                MPI_Gatherv(&gridData[gridPos], 2 * gridNumberOfPoints, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
        }
    }

    std::cout << "finalizando " << my_rank << std::endl;
    MPI_Finalize();
    return 0;
}