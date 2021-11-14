package main

import (
	"bufio"
	"fmt"
	"os"
	"math"
	"strconv"
	"strings"
)

var reader *bufio.Reader = bufio.NewReader(os.Stdin)
var writer *bufio.Writer = bufio.NewWriter(os.Stdout)

func printf(f string, a ...interface{}) { fmt.Fprintf(writer, f, a...) }
func scanf(f string, a ...interface{})  { fmt.Fscanf(reader, f, a...) }

func min_int(a int, b int) int { if a < b { return a } else { return b } }
func min_int64(a int64, b int64) int64 { if a < b { return a } else { return b } }

func max_int(a int, b int) int { if a > b { return a } else { return b } }
func max_int64(a int64, b int64) int64 { if a > b { return a } else { return b } }

func abs_int(a int) int { if a < 0 { return -a } else { return a } }
func abs_int64(a int64) int64 { if a < 0 { return -a } else { return a } }

const CAR_RENT_PRICE float64 = 10.0
const CAR_MOVE_PRICE float64 = 2.0

const RENTAL_REQ_FIRST_LOC = 3.0
const MAX_POISSON_REQS_FIRST_LOC = 12

const RENTAL_REQ_SECOND_LOC = 4.0
const MAX_POISSON_REQS_SECOND_LOC = 14

const RETURN_REQ_FIRST_LOC = 3.0
const MAX_POISSON_RETURNS_FIRST_LOC = 12

const RETURN_REQ_SECOND_LOC = 2.0
const MAX_POISSON_RETURNS_SECOND_LOC = 10

const MAX_CARS_AT_ANY_LOC = 20
const MAX_CARS_MOVED_IN_ONE_NIGHT = 5
// const NUMBER_OF_ACTIONS = 2 * MAX_CARS_MOVED_IN_ONE_NIGHT + 1

const GAMMA = 0.9

type State struct {
	i int
	j int
}


type PolicyAndValueFunction struct {
	policy [MAX_CARS_AT_ANY_LOC + 1][MAX_CARS_AT_ANY_LOC + 1]int
	value_function [MAX_CARS_AT_ANY_LOC + 1][MAX_CARS_AT_ANY_LOC + 1]float64
}


func (s State) String() string {
	return fmt.Sprintf("(%d %d)", s.i, s.j)
}


func factorial(k int64) int64 {
	if k == 0 {
		return 1
	}
	
	var r int64 = 1
	var i int64 
	for i = 1; i <= k; i++ {
		r = r * i
	}

	return r
}


func poisson(lambda float64, k float64) float64 {
	return ( math.Exp(-lambda) * math.Pow(lambda, k) ) / float64(factorial(int64(k)))
}


func generate_poisson_array(lambda float64, size int) []float64 {

	poisson_array := make([]float64, 0)
	for i := 0; i <= size; i++ {
		poisson_array = append(poisson_array, poisson(lambda, float64(i)))
	}

	return poisson_array
}


func generate_states() []State {

	states := make([]State, 0) 

	for i := 0; i <= MAX_CARS_AT_ANY_LOC; i++ {
		for j := 0; j <= MAX_CARS_AT_ANY_LOC; j++ {
			states = append(states, State{i, j})
		}
	}

	return states
}


func generate_actions() []int {

	actions := make([]int, 0)

	for i := -MAX_CARS_MOVED_IN_ONE_NIGHT; i <= MAX_CARS_MOVED_IN_ONE_NIGHT; i++ {
		actions = append(actions, i)
	}

	return actions
}


func approximate_state_value_function(state State,
									  value *[MAX_CARS_AT_ANY_LOC+1][MAX_CARS_AT_ANY_LOC+1]float64,
									  action int,
									  gamma float64,
									  poisson_first_rental_req []float64,
									  poisson_second_rental_req []float64,
									  poisson_first_rental_ret []float64,
									  poisson_second_rental_ret []float64) float64 {

	var r float64 = 0.0

	r += float64( abs_int(action) ) * CAR_MOVE_PRICE

    var cars_loc_one_morning int = min_int(max_int(state.i - action, 0), MAX_CARS_AT_ANY_LOC)
    var cars_loc_two_morning int = min_int(max_int(state.j - action, 0), MAX_CARS_AT_ANY_LOC)

	for rented_one := 0; rented_one <= MAX_POISSON_REQS_FIRST_LOC; rented_one++ {
		for rented_two := 0; rented_two <= MAX_POISSON_REQS_SECOND_LOC; rented_two++ {

            var actual_rented_cars_one int = min_int(cars_loc_one_morning, rented_one)
            var actual_rented_cars_two int = min_int(cars_loc_two_morning, rented_two)

			var earnings float64 = float64(actual_rented_cars_one + actual_rented_cars_two) * CAR_RENT_PRICE

			for returned_one := 0; returned_one <= MAX_POISSON_RETURNS_FIRST_LOC; returned_one++ {
				for returned_two := 0; returned_two <= MAX_POISSON_RETURNS_SECOND_LOC; returned_two++ {

					// fmt.Println("-> ", rented_one, " ", rented_two, " ", returned_one, " ", returned_two)

                    var cars_at_eob_one int = min_int(cars_loc_one_morning - actual_rented_cars_one + returned_one, MAX_CARS_AT_ANY_LOC)
                    var cars_at_eob_two int = min_int(cars_loc_one_morning - actual_rented_cars_one + returned_one, MAX_CARS_AT_ANY_LOC)

					var p float64 = poisson_first_rental_req[rented_one] *
									poisson_second_rental_req[rented_two] *
									poisson_first_rental_ret[returned_one] *
									poisson_second_rental_ret[returned_two]

					r = r + p * (earnings + gamma * value[cars_at_eob_one][cars_at_eob_two])
				}
			}
		}
	}

	// fmt.Println("r = ", r)
	return r
}


func evaluate_policy(value *[MAX_CARS_AT_ANY_LOC+1][MAX_CARS_AT_ANY_LOC+1]float64,
					 policy *[MAX_CARS_AT_ANY_LOC+1][MAX_CARS_AT_ANY_LOC+1]int,
					 actions []int,
					 states []State,
					 gamma float64,
					 theta float64,
					 poisson_first_rental_req []float64,
					 poisson_second_rental_req []float64,
					 poisson_first_rental_ret []float64,
					 poisson_second_rental_ret []float64) {

	for true {

		var delta float64 = 0.0

		for m := 0; m < len(states); m++ {

			// fmt.Println("m: ", m, " len(states): ", len(states))

			var i int = states[m].i 
			var j int = states[m].j
			var action int = policy[i][j]

			var v float64 = value[i][j]
			value[i][j] = approximate_state_value_function(states[m],
														   value,
														   action,
														   gamma, 
														   poisson_first_rental_req,
														   poisson_second_rental_req,
													       poisson_first_rental_ret,
														   poisson_second_rental_ret)
			delta = math.Max(delta, math.Abs(v - value[i][j]))

		}

		if delta < theta {
			fmt.Println("Delta = ", delta, " converged - breaking...")
			break
		}

	}
}


func policy_improvement(value *[MAX_CARS_AT_ANY_LOC+1][MAX_CARS_AT_ANY_LOC+1]float64,
						policy *[MAX_CARS_AT_ANY_LOC+1][MAX_CARS_AT_ANY_LOC+1]int,
						states []State,
						gamma float64,
						poisson_first_rental_req []float64,
						poisson_second_rental_req []float64,
						poisson_first_rental_ret []float64,
						poisson_second_rental_ret []float64) bool {

	var is_policy_stable bool = true

	for m := 0; m < len(states); m++ {

		var i int = states[m].i 
		var j int = states[m].j

		var old_action int = policy[i][j]

		var max_action_value float64 = -math.MaxFloat64
		var max_action int = 0

		var stop_action int = min_int(i,MAX_CARS_MOVED_IN_ONE_NIGHT)
		var start_action int = -min_int(j, MAX_CARS_MOVED_IN_ONE_NIGHT)

		for action := start_action; action <= stop_action; action++ {

			// if abs_int(action) > j || abs_int(action) > i {
			// 	// If action is e.g. -5 and j = 3 then we cannot execute the action since there are only 3 cars
			// 	// and we would like to move 5 which is not possible.
			// 	// fmt.Println("Oimiting action = ", action)
			// 	continue
			// }

			// fmt.Println("action = ", action)
			var r float64 = approximate_state_value_function(states[m],
															 value,
														 	 action,
															 gamma, 
															 poisson_first_rental_req,
															 poisson_second_rental_req,
															 poisson_first_rental_ret,
															 poisson_second_rental_ret)

			if r > max_action_value {
				max_action_value = r
				max_action = action
			}


		}

		policy[i][j] = max_action

		if old_action != policy[i][j] {
			is_policy_stable = false
		}


	}

	return is_policy_stable
}


func policy_iteration(value *[MAX_CARS_AT_ANY_LOC+1][MAX_CARS_AT_ANY_LOC+1]float64,
					  policy *[MAX_CARS_AT_ANY_LOC+1][MAX_CARS_AT_ANY_LOC+1]int,
					  actions []int,
					  states []State,
					  gamma float64,
					  theta float64,
					  poisson_first_rental_req []float64,
					  poisson_second_rental_req []float64,
					  poisson_first_rental_ret []float64,
					  poisson_second_rental_ret []float64) []PolicyAndValueFunction {

	pvf := make([]PolicyAndValueFunction, 0)

	var iter int = 1
	for true {

		fmt.Println("Iteration: ", iter)
		evaluate_policy(value,
						policy,
						actions,
						states,
						gamma,
						theta,
						poisson_first_rental_req,
						poisson_second_rental_req,
						poisson_first_rental_ret,
						poisson_second_rental_ret)



		var is_policy_stable bool = policy_improvement(value,
													   policy,
							                           states,
							                           gamma,
													   poisson_first_rental_req,
													   poisson_second_rental_req,
													   poisson_first_rental_ret,
													   poisson_second_rental_ret)

		pvf = append(pvf, PolicyAndValueFunction{*policy, *value})

		if is_policy_stable == true {
			break
		}
		iter++
	}

	return pvf
}


func save_2d_int_array_to_file(filename string,
							   array_2d *[MAX_CARS_AT_ANY_LOC+1][MAX_CARS_AT_ANY_LOC+1]int) {

	f, err := os.Create(filename)
    if err != nil {
        fmt.Printf("error creating file: %v", err)
        return
    }
    defer f.Close()
    for i := 0; i < MAX_CARS_AT_ANY_LOC+1; i++ {
		for j := 0; j < MAX_CARS_AT_ANY_LOC+1; j++ {

			_, err = f.WriteString(fmt.Sprintf("%d ", array_2d[i][j]))
			if err != nil {
				fmt.Printf("error writing string: %v", err)
			}
		}
		_, err = f.WriteString(fmt.Sprintf("\n"))
		if err != nil {
			fmt.Printf("error writing string: %v", err)
		}
	}
}

func save_2d_float64_array_to_file(filename string,
							       array_2d *[MAX_CARS_AT_ANY_LOC+1][MAX_CARS_AT_ANY_LOC+1]float64) {

	f, err := os.Create(filename)
    if err != nil {
        fmt.Printf("error creating file: %v", err)
        return
    }
    defer f.Close()
    for i := 0; i < MAX_CARS_AT_ANY_LOC+1; i++ {
		for j := 0; j < MAX_CARS_AT_ANY_LOC+1; j++ {

			_, err = f.WriteString(fmt.Sprintf("%f ", array_2d[i][j]))
			if err != nil {
				fmt.Printf("error writing string: %v", err)
			}
		}
		_, err = f.WriteString(fmt.Sprintf("\n"))
		if err != nil {
			fmt.Printf("error writing string: %v", err)
		}
	}
}

func print_slice_of_int(s []float64) {
	for i := 0; i < len(s); i++ {
		fmt.Println("i: ", i, " -> ", s[i])
	}
}


func main() {

	poisson_first_rental_req := generate_poisson_array(RENTAL_REQ_FIRST_LOC, MAX_POISSON_REQS_FIRST_LOC + 1)
	poisson_second_rental_req := generate_poisson_array(RENTAL_REQ_SECOND_LOC, MAX_POISSON_REQS_SECOND_LOC + 1)
	poisson_first_rental_ret := generate_poisson_array(RETURN_REQ_FIRST_LOC, MAX_POISSON_RETURNS_FIRST_LOC + 1)
	poisson_second_rental_ret := generate_poisson_array(RETURN_REQ_SECOND_LOC, MAX_POISSON_RETURNS_SECOND_LOC + 1)

	// print_slice_of_int(poisson_first_rental_req)
	// print_slice_of_int(poisson_second_rental_req)
	// print_slice_of_int(poisson_first_rental_ret)
	// print_slice_of_int(poisson_second_rental_ret)

	states := generate_states()
	actions := generate_actions()


	var policy [MAX_CARS_AT_ANY_LOC+1][MAX_CARS_AT_ANY_LOC+1]int
	var value [MAX_CARS_AT_ANY_LOC+1][MAX_CARS_AT_ANY_LOC+1]float64

	// fmt.Println(policy[0])
	// fmt.Println(value[0])

	var theta float64 = 1.0/10.0
	var gamma float64 = 0.9
	// evaluate_policy(&value,
	// 				&policy,
	// 				actions,
	// 				states,
	// 				gamma,
	// 				theta,
	// 				poisson_first_rental_req,
	// 				poisson_second_rental_req,
	// 				poisson_first_rental_ret,
	// 				poisson_second_rental_ret)

	// generate_poisson_array(lambda float64, size int)

	pvf := policy_iteration(&value,
					        &policy,
					        actions,
							states,
							gamma,
							theta,
							poisson_first_rental_req,
							poisson_second_rental_req,
							poisson_first_rental_ret,
							poisson_second_rental_ret)

	for i := 0; i < len(pvf); i++ {

		filename_policy := strings.Join([]string{"policy_iter_", strconv.Itoa(i), ".txt"}, "");
		save_2d_int_array_to_file(filename_policy, &pvf[i].policy)

		filename_value := strings.Join([]string{"value_iter_", strconv.Itoa(i), ".txt"}, "");
		save_2d_float64_array_to_file(filename_value, &pvf[i].value_function)
	}


}