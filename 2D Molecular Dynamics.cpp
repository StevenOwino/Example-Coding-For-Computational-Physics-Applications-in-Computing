// The simulation uses reduced Lennard–Jones units, with particle mass, σ, and ϵ set to 1. If the particles become unstable, reduce DT(Molecular Dynamics time step); integration accuracy depends on the timestep, and interaction strength.

#include <SFML/Graphics.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

struct Vec2 {
    double x = 0.0;
    double y = 0.0;

    Vec2 operator+(const Vec2& b) const { return {x + b.x, y + b.y}; }
    Vec2 operator-(const Vec2& b) const { return {x - b.x, y - b.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    Vec2& operator+=(const Vec2& b) {
        x += b.x;
        y += b.y;
        return *this;
    }
};

constexpr unsigned int WINDOW_W = 900;
constexpr unsigned int WINDOW_H = 700;

constexpr int NX = 8;
constexpr int NY = 8;
constexpr int N = NX * NY;

constexpr double BOX = 20.0;       // Square simulation-box side length
constexpr double EPSILON = 1.0;    // Lennard-Jones energy scale
constexpr double SIGMA = 1.0;      // Lennard-Jones length scale
constexpr double CUTOFF = 2.5;     // Interaction cutoff
constexpr double DT = 0.002;       // MD time step
constexpr int STEPS_PER_FRAME = 10;

constexpr double INITIAL_TEMPERATURE = 0.7;

constexpr float PIXEL_MARGIN = 35.0f;
constexpr float PIXEL_BOX =
    static_cast<float>(WINDOW_H) - 2.0f * PIXEL_MARGIN;

double dot(const Vec2& a, const Vec2& b) {
    return a.x * b.x + a.y * b.y;
}

double lengthSquared(const Vec2& a) {
    return dot(a, a);
}

void applyPeriodicBoundaries(Vec2& p) {
    p.x -= BOX * std::floor(p.x / BOX);
    p.y -= BOX * std::floor(p.y / BOX);
}

Vec2 minimumImage(Vec2 d) {
    d.x -= BOX * std::round(d.x / BOX);
    d.y -= BOX * std::round(d.y / BOX);
    return d;
}

// Radial Lennard-Jones force, with a force shift so it reaches zero at cutoff.
double lennardJonesRadialForce(double r) {
    const double sr = SIGMA / r;
    const double sr2 = sr * sr;
    const double sr6 = sr2 * sr2 * sr2;
    const double sr12 = sr6 * sr6;
    return 24.0 * EPSILON * (2.0 * sr12 - sr6) / r;
}

const double FORCE_AT_CUTOFF = lennardJonesRadialForce(CUTOFF);

void calculateForces(const std::vector<Vec2>& positions,
                     std::vector<Vec2>& forces) {
    std::fill(forces.begin(), forces.end(), Vec2{});

    const double cutoff2 = CUTOFF * CUTOFF;

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            const Vec2 displacement =
                minimumImage(positions[i] - positions[j]);
            const double r2 = lengthSquared(displacement);

            if (r2 >= cutoff2 || r2 < 1e-12) {
                continue;
            }

            const double r = std::sqrt(r2);
            const double radialForce =
                lennardJonesRadialForce(r) - FORCE_AT_CUTOFF;

            // Force vector on particle i.
            const Vec2 force = displacement * (radialForce / r);
            forces[i] += force;
            forces[j] += force * -1.0;
        }
    }
}

sf::Vector2f toScreen(const Vec2& p) {
    return {
        PIXEL_MARGIN + static_cast<float>(p.x / BOX) * PIXEL_BOX,
        PIXEL_MARGIN + static_cast<float>(p.y / BOX) * PIXEL_BOX
    };
}

sf::Color speedColor(double speed) {
    // Map speed to a blue-to-red color range.
    const double t = std::clamp(speed / 2.5, 0.0, 1.0);
    const auto red = static_cast<sf::Uint8>(255.0 * t);
    const auto blue = static_cast<sf::Uint8>(255.0 * (1.0 - t));
    return sf::Color(red, 80, blue);
}

int main() {
    std::mt19937 rng(12345);
    std::normal_distribution<double> velocityDistribution(
        0.0, std::sqrt(INITIAL_TEMPERATURE)
    );

    std::vector<Vec2> positions(N);
    std::vector<Vec2> velocities(N);
    std::vector<Vec2> forces(N);

    // Place particles on a square lattice.
    const double spacing = BOX / NX;
    for (int iy = 0; iy < NY; ++iy) {
        for (int ix = 0; ix < NX; ++ix) {
            const int i = iy * NX + ix;
            positions[i] = {
                (ix + 0.5) * spacing,
                (iy + 0.5) * spacing
            };
        }
    }

    // Assign random velocities.
    Vec2 meanVelocity{};
    for (Vec2& v : velocities) {
        v = {velocityDistribution(rng), velocityDistribution(rng)};
        meanVelocity += v;
    }

    // Remove net center-of-mass motion.
    meanVelocity = meanVelocity * (1.0 / N);
    for (Vec2& v : velocities) {
        v = v - meanVelocity;
    }

    calculateForces(positions, forces);

    sf::RenderWindow window(
        sf::VideoMode(WINDOW_W, WINDOW_H),
        "2D Molecular Dynamics (Lennard-Jones)"
    );
    window.setFramerateLimit(60);

    sf::RectangleShape boxOutline(
        sf::Vector2f(PIXEL_BOX, PIXEL_BOX)
    );
    boxOutline.setPosition(PIXEL_MARGIN, PIXEL_MARGIN);
    boxOutline.setFillColor(sf::Color::Transparent);
    boxOutline.setOutlineThickness(2.0f);
    boxOutline.setOutlineColor(sf::Color(180, 180, 180));

    std::vector<sf::CircleShape> particles(N);
    for (auto& particle : particles) {
        particle.setRadius(5.0f);
        particle.setOrigin(5.0f, 5.0f);
    }

    bool paused = false;

    while (window.isOpen()) {
        sf::Event event{};
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }

            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Space) {
                    paused = !paused;
                }
                if (event.key.code == sf::Keyboard::R) {
                    // Restart the application to reset the simulation.
                    return main();
                }
            }
        }

        if (!paused) {
            for (int step = 0; step < STEPS_PER_FRAME; ++step) {
                // Velocity-Verlet position update.
                for (int i = 0; i < N; ++i) {
                    positions[i] += velocities[i] * DT
                                  + forces[i] * (0.5 * DT * DT);
                    applyPeriodicBoundaries(positions[i]);
                }

                std::vector<Vec2> newForces(N);
                calculateForces(positions, newForces);

                // Velocity-Verlet velocity update.
                for (int i = 0; i < N; ++i) {
                    velocities[i] +=
                        (forces[i] + newForces[i]) * (0.5 * DT);
                }

                forces.swap(newForces);
            }
        }

        window.clear(sf::Color(18, 20, 28));
        window.draw(boxOutline);

        for (int i = 0; i < N; ++i) {
            particles[i].setPosition(toScreen(positions[i]));
            particles[i].setFillColor(
                speedColor(std::sqrt(lengthSquared(velocities[i])))
            );
            window.draw(particles[i]);
        }

        window.display();
    }

    return 0;
}
