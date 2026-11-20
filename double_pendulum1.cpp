#include <SFML/Graphics.hpp>
#include <array>
#include <cmath>
#include <iostream>

constexpr double pi = 3.14159265358979323846;
constexpr double g = 9.81;

constexpr double m1 = 1.0;
constexpr double m2 = 1.0;
constexpr double l1 = 1.0;
constexpr double l2 = 1.0;

using State = std::array<double, 4>; // theta1, omega1, theta2, omega2

State derivative(const State& s) {
    const double theta1 = s[0];
    const double omega1 = s[1];
    const double theta2 = s[2];
    const double omega2 = s[3];

    const double delta = theta1 - theta2;
    const double c = std::cos(delta);
    const double sn = std::sin(delta);

    // Matrix coefficients for the angular accelerations
    const double a = (m1 + m2) * l1 * l1;
    const double b = m2 * l1 * l2 * c;
    const double d = m2 * l2 * l2;

    // Right-hand side
    const double r1 =
        -m2 * l1 * l2 * sn * omega2 * omega2
        - (m1 + m2) * g * l1 * std::sin(theta1);

    const double r2 =
         m2 * l1 * l2 * sn * omega1 * omega1
        - m2 * g * l2 * std::sin(theta2);

    const double det = a * d - b * b;
    const double alpha1 = (d * r1 - b * r2) / det;
    const double alpha2 = (a * r2 - b * r1) / det;

    return {omega1, alpha1, omega2, alpha2};
}

State addScaled(const State& s, const State& k, double scale) {
    State result{};
    for (int i = 0; i < 4; ++i) {
        result[i] = s[i] + scale * k[i];
    }
    return result;
}

State rk4Step(const State& s, double dt) {
    const State k1 = derivative(s);
    const State k2 = derivative(addScaled(s, k1, dt / 2.0));
    const State k3 = derivative(addScaled(s, k2, dt / 2.0));
    const State k4 = derivative(addScaled(s, k3, dt));

    State next{};
    for (int i = 0; i < 4; ++i) {
        next[i] = s[i] + (dt / 6.0) *
            (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    return next;
}

int main() {
    // Initial angles are measured from the downward vertical, in radians.
    State state = {pi / 2.0, 0.0, pi / 2.0 + 0.05, 0.0};

    sf::RenderWindow window(
        sf::VideoMode(900, 700),
        "Double Pendulum"
    );
    window.setFramerateLimit(60);

    const double dt = 1.0 / 600.0;
    const float scale = 180.0f; // pixels per metre
    const sf::Vector2f pivot(450.0f, 200.0f);

    sf::VertexArray trail(sf::LineStrip);
    unsigned int frameCount = 0;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        // Advance the physics by several small steps per rendered frame.
        for (int i = 0; i < 10; ++i) {
            state = rk4Step(state, dt);
        }

        const double theta1 = state[0];
        const double theta2 = state[2];

        // Convert angles to screen coordinates.
        const sf::Vector2f p1(
            pivot.x + scale * static_cast<float>(l1 * std::sin(theta1)),
            pivot.y + scale * static_cast<float>(l1 * std::cos(theta1))
        );

        const sf::Vector2f p2(
            p1.x + scale * static_cast<float>(l2 * std::sin(theta2)),
            p1.y + scale * static_cast<float>(l2 * std::cos(theta2))
        );

        // Add a trail point occasionally to keep the trail manageable.
        if (frameCount++ % 2 == 0) {
            trail.append(sf::Vertex(p2, sf::Color(220, 80, 70)));
        }

        sf::Vertex rods[] = {
            sf::Vertex(pivot, sf::Color::White),
            sf::Vertex(p1, sf::Color::White),
            sf::Vertex(p1, sf::Color::White),
            sf::Vertex(p2, sf::Color::White)
        };

        sf::CircleShape bob1(12.0f);
        bob1.setOrigin(12.0f, 12.0f);
        bob1.setPosition(p1);
        bob1.setFillColor(sf::Color(80, 150, 255));

        sf::CircleShape bob2(12.0f);
        bob2.setOrigin(12.0f, 12.0f);
        bob2.setPosition(p2);
        bob2.setFillColor(sf::Color(255, 180, 60));

        sf::CircleShape pivotMarker(6.0f);
        pivotMarker.setOrigin(6.0f, 6.0f);
        pivotMarker.setPosition(pivot);
        pivotMarker.setFillColor(sf::Color::White);

        window.clear(sf::Color(25, 25, 35));
        window.draw(trail);
        window.draw(rods, 4, sf::Lines);
        window.draw(pivotMarker);
        window.draw(bob1);
        window.draw(bob2);
        window.display();
    }

    return 0;
}
