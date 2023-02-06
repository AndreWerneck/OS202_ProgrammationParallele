#include <SDL/SDL.h>
#include <numeric>
#include <complex>

int main() {
    SDL_Init(SDL_INIT_EVERYTHING);
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_CreateWindowAndRenderer(2000,2000, 0, &window, &renderer);
    SDL_RenderSetScale(renderer, 2, 2);


    for (double i=0.0;i < 1.0;i+=0.01) {
        for (double j=0.0;j < 1.0;j+=0.01) {

        }
    }

    SDL_RenderPresent(renderer);
    SDL_Delay(100000);

    return 0;
}