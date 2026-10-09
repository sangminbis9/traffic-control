"""Cache current TraCI measurements using subscriptions without changing SUMO.

The original state provider still constructs all observations. Only its data
accessors are served from TraCI's values for the same simulation tick.
"""

from __future__ import annotations

import traci.constants as tc


class VehicleDomain:
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner.backend.vehicle, name)

    def getIDList(self):
        return self.owner.vehicle_ids

    def _get(self, vehicle, variable, getter):
        result = self.owner.vehicle_values.get(vehicle, {})
        if variable in result:
            return result[variable]
        return getattr(self.owner.backend.vehicle, getter)(vehicle)

    def getLaneID(self, vehicle):
        return self._get(vehicle, tc.VAR_LANE_ID, 'getLaneID')

    def getAccumulatedWaitingTime(self, vehicle):
        return self._get(vehicle, tc.VAR_ACCUMULATED_WAITING_TIME, 'getAccumulatedWaitingTime')

    def getSpeed(self, vehicle):
        return self._get(vehicle, tc.VAR_SPEED, 'getSpeed')

    def getLanePosition(self, vehicle):
        return self._get(vehicle, tc.VAR_LANEPOSITION, 'getLanePosition')


class LaneDomain:
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner.backend.lane, name)

    def getLength(self, lane):
        if lane in self.owner.lane_lengths:
            return self.owner.lane_lengths[lane]
        return self.owner.backend.lane.getLength(lane)

    def getLastStepHaltingNumber(self, lane):
        values = self.owner.lane_values.get(lane, {})
        if tc.LAST_STEP_VEHICLE_HALTING_NUMBER in values:
            return values[tc.LAST_STEP_VEHICLE_HALTING_NUMBER]
        return self.owner.backend.lane.getLastStepHaltingNumber(lane)


class SubscriptionConnection:
    def __init__(self, backend):
        self.backend = backend
        self.vehicle = VehicleDomain(self)
        self.lane = LaneDomain(self)
        self.lane_lengths = {}
        self.vehicle_ids = ()
        self.vehicle_values = {}
        self.lane_values = {}
        self.variables = (tc.VAR_LANE_ID, tc.VAR_ACCUMULATED_WAITING_TIME,
                          tc.VAR_SPEED, tc.VAR_LANEPOSITION)
        for direction in ('N', 'S', 'E', 'W'):
            for index in range(3):
                lane = f'{direction}_in_{index}'
                self.lane_lengths[lane] = backend.lane.getLength(lane)
                backend.lane.subscribe(lane, (tc.LAST_STEP_VEHICLE_HALTING_NUMBER,))
        self.refresh()

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def refresh(self):
        self.vehicle_ids = tuple(self.backend.vehicle.getIDList())
        results = self.backend.vehicle.getAllSubscriptionResults()
        for vehicle in self.vehicle_ids:
            if vehicle not in results:
                self.backend.vehicle.subscribe(vehicle, self.variables)
        self.vehicle_values = self.backend.vehicle.getAllSubscriptionResults()
        self.lane_values = self.backend.lane.getAllSubscriptionResults()

    def simulationStep(self, *args, **kwargs):
        result = self.backend.simulationStep(*args, **kwargs)
        self.refresh()
        return result
