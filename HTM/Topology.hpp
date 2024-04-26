/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ----------------------------------------------------------------------
 */

 /** @file
  * Topology helpers
  */

#ifndef NTA_TOPOLOGY_HPP
#define NTA_TOPOLOGY_HPP

#include <vector>
#include <stdexcept>
#include "Type.hpp"

namespace topology {
    template <typename T>
    void sample(T population[], UInt nPopulation, T choices[],
        UInt nChoices) {
        if (nChoices == 0) {
            return;
        }
        if (nChoices > nPopulation) {
            throw std::runtime_error("population size must be greater than number of choices");
        }
        UInt nextChoice = 0;
        for (UInt i = 0; i < nPopulation; ++i) {
            if (rand() % (nPopulation - i) < (nChoices - nextChoice)) {
                choices[nextChoice] = population[i];
                ++nextChoice;
                if (nextChoice == nChoices) {
                    break;
                }
            }
        }
    }
    class CoordinateConverter2D {

    public:
        CoordinateConverter2D(UInt nrows, UInt ncols)
            : // TODO param nrows is unused
            ncols_(ncols) {}
        UInt toRow(UInt index) { return index / ncols_; };
        UInt toCol(UInt index) { return index % ncols_; };
        UInt toIndex(UInt row, UInt col) { return row * ncols_ + col; };

    private:
        UInt ncols_;
    };

    class CoordinateConverterND {

    public:
        CoordinateConverterND(std::vector<UInt>& dimensions) {
            dimensions_ = dimensions;
            UInt b = 1;
            for (Int i = (Int)dimensions.size() - 1; i >= 0; i--) {
                bounds_.insert(bounds_.begin(), b);
                b *= dimensions[i];
            }
        }

        void toCoord(UInt index, std::vector<UInt>& coord) {
            coord.clear();
            for (UInt i = 0; i < bounds_.size(); i++) {
                coord.push_back((index / bounds_[i]) % dimensions_[i]);
            }
        };

        UInt toIndex(std::vector<UInt>& coord) {
            UInt index = 0;
            for (UInt i = 0; i < coord.size(); i++) {
                index += coord[i] * bounds_[i];
            }
            return index;
        };

    private:
        std::vector<UInt> dimensions_;
        std::vector<UInt> bounds_;
    };
    /**
        * Translate an index into coordinates, using the given coordinate system.
        *
        * @param index
        * The index of the point. The coordinates are expressed as a single index
        * by using the dimensions as a mixed radix definition. For example, in
        * dimensions 42x10, the point [1, 4] is index 1*420 + 4*10 = 460.
        *
        * @param dimensions
        * The coordinate system.
        *
        * @returns
        * A vector of coordinates of length dimensions.size().
        */
    std::vector<UInt> coordinatesFromIndex(UInt index,
        const std::vector<UInt>& dimensions);

    /**
        * Translate coordinates into an index, using the given coordinate system.
        *
        * @param coordinates
        * A vector of coordinates of length dimensions.size().
        *
        * @param dimensions
        * The coordinate system.
        *
        * @returns
        * The index of the point. The coordinates are expressed as a single index
        * by using the dimensions as a mixed radix definition. For example, in
        * dimensions 42x10, the point [1, 4] is index 1*420 + 4*10 = 460.
        */
    UInt indexFromCoordinates(const std::vector<UInt>& coordinates,
        const std::vector<UInt>& dimensions);

    /**
        * A class that lets you iterate over all points within the neighborhood
        * of a point.
        *
        * Usage:
        *   UInt center = 42;
        *   for (UInt neighbor : Neighborhood(center, 10, {100, 100}))
        *   {
        *     if (neighbor == center)
        *     {
        *       // Note that the center is included in the neighborhood!
        *     }
        *     else
        *     {
        *       // Do something with the neighbor.
        *     }
        *   }
        *
        * A point's neighborhood is the n-dimensional hypercube with sides
        * ranging [center - radius, center + radius], inclusive. For example,
        * if there are two dimensions and the radius is 3, the neighborhood is
        * 6x6. Neighborhoods are truncated when they are near an edge.
        *
        * Dimensions aren't copied -- a reference is saved. Make sure the
        * dimensions don't get overwritten while this Neighborhood instance
        * exists.
        *
        * This is designed to be fast. It walks the list of points in the
        * neighborhood without ever creating a list of points.
        *
        * This still could be faster. Because it handles an arbitrary number of
        * dimensions, it has to allocate vectors. It would be faster to have a
        * Neighborhood1D, Neighborhood2D, etc., so that all computation could
        * occur on the stack, but this would put a burden on callers to handle
        * different dimensions counts. Or it would require using polymorphism,
        * using pointers/references and putting the Neighborhood on the heap,
        * which defeats the purpose of avoiding the vector allocations.
        *
        * @param centerIndex
        * The center of this neighborhood. The coordinates are expressed as a
        * single index by using the dimensions as a mixed radix definition. For
        * example, in dimensions 42x10, the point [1, 4] is index 1*420 + 4*10 =
        * 460.
        *
        * @param radius
        * The radius of this neighborhood about the centerIndex.
        *
        * @param dimensions
        * The dimensions of the world outside this neighborhood.
        *
        * @returns
        * An object which supports C++ range-based for loops. Each iteration of
        * the loop returns a point in the neighborhood. Each point is expressed
        * as a single index.
        */
    class Neighborhood {
    public:
        Neighborhood(UInt centerIndex, UInt radius,
            const std::vector<UInt>& dimensions);

        class Iterator {
        public:
            Iterator(const Neighborhood& neighborhood, bool end);
            bool operator!=(const Iterator& other) const;
            UInt operator*() const;
            const Iterator& operator++();

        private:
            void advance_();

            const Neighborhood& neighborhood_;
            std::vector<Int> offset_;
            bool finished_;
        };

        Iterator begin() const;
        Iterator end() const;

    private:
        const std::vector<UInt> centerPosition_;
        const std::vector<UInt>& dimensions_;
        const UInt radius_;
    };

    /**
        * Like the Neighborhood class, except that the neighborhood isn't
        * truncated when it's near an edge. It wraps around to the other side.
        *
        * @param centerIndex
        * The center of this neighborhood. The coordinates are expressed as a
        * single index by using the dimensions as a mixed radix definition. For
        * example, in dimensions 42x10, the point [1, 4] is index 1*420 + 4*10 =
        * 460.
        *
        * @param radius
        * The radius of this neighborhood about the centerIndex.
        *
        * @param dimensions
        * The dimensions of the world outside this neighborhood.
        *
        * @returns
        * An object which supports C++ range-based for loops. Each iteration of
        * the loop returns a point in the neighborhood. Each point is expressed
        * as a single index.
        */
    class WrappingNeighborhood {
    public:
        WrappingNeighborhood(UInt centerIndex, UInt radius,
            const std::vector<UInt>& dimensions);

        class Iterator {
        public:
            Iterator(const WrappingNeighborhood& neighborhood, bool end);
            bool operator!=(const Iterator& other) const;
            UInt operator*() const;
            const Iterator& operator++();

        private:
            void advance_();

            const WrappingNeighborhood& neighborhood_;
            std::vector<Int> offset_;
            bool finished_;
        };

        Iterator begin() const;
        Iterator end() const;

    private:
        const std::vector<UInt> centerPosition_;
        const std::vector<UInt>& dimensions_;
        const UInt radius_;
    };

} // end namespace topology

#endif // NTA_TOPOLOGY_HPP
