#include <fstream>
#include <iostream>
#include <string>
#include <rapidxml.hpp>
#include <helper_functions.h>

#include "Problem/IO/input-settings.hh"
#include "Problem/IO/xml.hh"
#include "Kernel/globals.hh"
#include "Utilities/utils.hh"
#include "preferences.hh"


InputSettings::InputSettings(int argc, char* argv[])
  : p_preprocess("standard"), p_resolution_method("DPOP"), 
   p_inner_message_cost(0), p_outer_message_cost(0),
   p_NCCCs_limit(0), p_memory_limit(0), 
    p_simulated_timeout_ms(0), p_wallclock_timeout_ms(0),
    p_elected_root( 0 ), p_heuristic( 2 ), p_max_gpu_size( 6000 )
{

	if (preferences::importPseudoTree) {

		if (existsSavedPseudoTree(argv[ 1 ])) {
			loadPseudoTree(argv[ 1 ]);
		}
	} else {
		if (checkCmdLineFlag(argc, (const char **) argv, "root")) {
			p_elected_root = getCmdLineArgumentInt(argc, (const char **) argv, "root");
		}
		if (checkCmdLineFlag(argc, (const char **) argv, "heur")) {
			p_heuristic = getCmdLineArgumentInt(argc, (const char **) argv, "heur");
		}
	}

  // std::cout << "Sat Elected Root = " << p_elected_root << "\n";
  // checkparams(argc, argv);
  // p_filename = argv[ 2 ];  
  // import(p_filename);
}


void InputSettings::loadPseudoTree(std::string file) {
	file = file.substr(0, file.find_last_of("."));
	file += ".ptf";// pseudo-tree format
	std::string sep = " ";
	std::ifstream ifs;
	ifs.open(file.c_str(), std::ifstream::in);
	int size;
	std::string line;
	getline(ifs, line); // skip first line (root and heuristics)
	std::stringstream data(line);
	data >> size >> p_elected_root >> p_heuristic;

	ifs.close();
}


bool InputSettings::existsSavedPseudoTree(std::string file) {
	file = file.substr(0, file.find_last_of("."));
	file += ".ptf";// pseudo-tree format
	std::ifstream f(file.c_str());
	return f.good();
}

void InputSettings::import(std::string xml_file)
{
  std::string input_xml;
  std::string line;
  std::ifstream in( xml_file.c_str() );
  ASSERT (in.is_open(), "Error: cannot open the input file.");
  while (getline( in, line ))
    input_xml += line;
  
  // make a safe-to-modify copy of input_xml
  std::vector<char> xml_copy(input_xml.begin(), input_xml.end());
  xml_copy.push_back('\0');
  rapidxml::xml_document<> doc;
  doc.parse<rapidxml::parse_declaration_node | 
    rapidxml::parse_no_data_nodes>( &xml_copy[ 0 ] );
  rapidxml::xml_node<>* root = doc.first_node("settings");

  rapidxml::xml_node<>* xdcop = root->first_node("DCOPsolver");
  parseDCOPsolver(xdcop);
  
  rapidxml::xml_node<>* xmeasures = root->first_node("metrics");
  parseMetrics(xmeasures);
  
  rapidxml::xml_node<>* xtimeouts = root->first_node("timeouts");
  parseTimeouts(xtimeouts);
  
  in.close();
}


void InputSettings::parseDCOPsolver(rapidxml::xml_node<>* xdcop)
{
  p_preprocess = xdcop->first_attribute("preprocess")->value();
  p_resolution_method = xdcop->first_attribute("schema")->value();
  
  if(xdcop->first_attribute("parameters"))
  {
    std::string params = xdcop->first_attribute("parameters")->value();
    p_resolution_params = Utils::stringSplit(Utils::trim(params));
  }
  
  rapidxml::xml_node<>* xsolver = xdcop->first_node("solver");  
  do {
    std::string agent_name = xsolver->first_attribute("agent")->value();
    std::string strategy = xsolver->first_attribute("strategy")->value();
    std::vector<std::string> values = xml::getStringValues(xsolver);

    p_map_agent_search_strategy_params[ agent_name ]
      = std::make_pair(strategy, values);

    xsolver = xsolver->next_sibling();
  } while(xsolver);
}


void InputSettings::parseMetrics(rapidxml::xml_node<>* xmeasures)
{
  rapidxml::xml_node<>* xmeasure = xmeasures->first_node("metric");
  do {
    std::string name = xmeasure->first_attribute("name")->value();
    if(name.compare("innerMessageCost") == 0)
      p_inner_message_cost = atoi(xmeasure->value());
    if(name.compare("outerMessageCost") == 0)
      p_outer_message_cost = atoi(xmeasure->value());
  
    xmeasure = xmeasure->next_sibling();
  } while(xmeasure);  
}


void InputSettings::parseTimeouts(rapidxml::xml_node<>* xtimeouts)
{
  rapidxml::xml_node<>* xtimeout = xtimeouts->first_node("timeout");
  do {
    std::string name = xtimeout->first_attribute("metric")->value();
    if(name.compare("simulatedTime") == 0)
      p_simulated_timeout_ms = atoi(xtimeout->value());
    if(name.compare("wallclockTime") == 0)
      p_wallclock_timeout_ms = atoi(xtimeout->value());
    if(name.compare("NCCCs") == 0)
      p_NCCCs_limit = atoi(xtimeout->value());
    if(name.compare("memory") == 0)
      p_memory_limit = atoi(xtimeout->value());
  
    xtimeout = xtimeout->next_sibling();
  } while(xtimeout);  
}


InputSettings::solving_t InputSettings::boundarySolver()
{
  std::vector<std::string> empty;
  std::vector<std::string> gibbs(2,"1");

  if (p_resolution_method.compare("DGIBBS") == 0 ||
      p_resolution_method.compare("dgibbs") == 0)
    return std::make_pair("Gibbs", gibbs);

  if(p_preprocess.compare("tableau") == 0)
      return std::make_pair("Tableau", empty);
  else
      return std::make_pair("DFS", empty); 
}


std::string InputSettings::dump()
{
  std::string res;
  res += "DCOP solving method     : " + p_preprocess + " " + p_resolution_method + '\n';
  res += "Agents solving strategy : ";
  if (singlePrivateSolver())
  {
    res += privateSolver("*").first + " ";
    for (std::string param : privateSolver("*").second)
      res += param + " ";
  }
  else
  {
    res += '\n';
    for (auto& kv : p_map_agent_search_strategy_params)
    {
      res += kv.first + ": " + kv.second.first + " ";
      for (std::string param : kv.second.second)
        res += param + " ";
      res += '\n';
    }
  }
  res += '\n';
  res += "Message Costs           : inner(" 
      + std::to_string(p_inner_message_cost) + ")"
      + " outer("+ std::to_string(p_outer_message_cost) +")";
  return res;
}
























